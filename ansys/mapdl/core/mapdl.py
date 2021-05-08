"""Module to control interaction with MAPDL through Python"""
import time
import glob
import re
import os
import logging
import tempfile
from shutil import rmtree, copyfile
import weakref
import warnings
import pathlib

import numpy as np

from ansys.mapdl import core as pymapdl
from ansys.mapdl.core.mapdl_functions import _MapdlCommands
from ansys.mapdl.core.misc import (random_string, supress_logging,
                                   run_as_prep7, last_created)
from ansys.mapdl.core.element_commands import element_commands
from ansys.mapdl.core.errors import MapdlRuntimeError, MapdlInvalidRoutineError
from ansys.mapdl.core.plotting import general_plotter
from ansys.mapdl.core.post import PostProcessing


_PERMITTED_ERRORS = [
    r'(\*\*\* ERROR \*\*\*).*(?:[\r\n]+.*)+highly distorted.',
    r'(\*\*\* ERROR \*\*\*).*[\r\n]+.*is turning inside out.',
]

# test for png file
PNG_TEST = re.compile('WRITTEN TO FILE(.*).png')

VWRITE_REPLACEMENT = """
Cannot use *VWRITE directly as a command in MAPDL
service mode.  Instead, run it as ``non_interactive``.

For example:

with self.non_interactive:
    self.vwrite('%s(1)' % parm_name)
    self.run('(F20.12)')
"""

INVAL_COMMANDS = {'*VWR': VWRITE_REPLACEMENT,
                  '*CFO': 'Run CFOPEN as non_interactive',
                  '*CRE': 'Create a function within python or run as non_interactive',
                  '*END': 'Create a function within python or run as non_interactive',
                  '*IF': 'Use a python if or run as non_interactive'}

PLOT_COMMANDS = ['NPLO', 'EPLO', 'KPLO', 'LPLO', 'APLO', 'VPLO', 'PLNS', 'PLES']
MAX_COMMAND_LENGTH = 600  # actual is 640, but seems to fail above 620


def parse_to_short_cmd(command):
    """Takes any MAPDL command and returns the first 4 characters of
    the command

    Examples
    --------
    >>> parse_to_short_cmd('K,,1,0,0,')
    'K'

    >>> parse_to_short_cmd('VPLOT, ALL')
    'VPLO'
    """
    try:
        short_cmd = command.split(',')[0]
        return short_cmd[:4].upper()
    except:  # pragma: no cover
        return


def setup_logger(loglevel='INFO'):
    """Setup logger"""

    # return existing log if this function has already been called
    if hasattr(setup_logger, 'log'):
        setup_logger.log.setLevel(loglevel)
        ch = setup_logger.log.handlers[0]
        ch.setLevel(loglevel)
        return setup_logger.log

    # create logger
    log = logging.getLogger(__name__)
    log.setLevel(loglevel)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)

    # create formatter
    formatstr = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    formatter = logging.Formatter(formatstr)

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    log.addHandler(ch)

    # make persistent
    setup_logger.log = log

    return log


class _MapdlCore(_MapdlCommands):
    """Contains methods in common between all Mapdl subclasses"""

    def __init__(self, loglevel='DEBUG', use_vtk=True, log_apdl=False,
                 local=True, **start_parm):
        """Initialize connection with MAPDL. """
        self._show_matplotlib_figures = True  # for testing
        self._exited = False
        self._allow_ignore = False
        self._apdl_log = None
        self._store_commands = False
        self._stored_commands = []
        self._response = None
        self._use_vtk = use_vtk
        self._log_filehandler = None
        self._version = None  # cached version
        self._local = local
        self._jobname = start_parm.get('jobname', 'file')
        self._cleanup = True
        self._vget_arr_counter = 0
        self._start_parm = start_parm
        self._path = start_parm.get('run_location', None)
        self._ignore_errors = False

        self._log = setup_logger(loglevel.upper())
        self._log.debug('Logging set to %s', loglevel)
        self.non_interactive = self._non_interactive(self)

        from ansys.mapdl.core.parameters import Parameters
        self._parameters = Parameters(self)

        from ansys.mapdl.core.solution import Solution
        self._solution = Solution(self)

        if log_apdl:
            filename = os.path.join(self.directory, 'log.inp')
            self.open_apdl_log(filename, mode=log_apdl)

        self._post = PostProcessing(self)

    @property
    def solution(self):
        """Solution parameters of MAPDL.

        Examples
        --------
        Check if a solution has converged.

        >>> mapdl.solution.converged
        """
        if self._exited:
            raise RuntimeError('MAPDL exited.')
        return self._solution

    @property
    def _distributed(self):
        """Is a distributed analysis"""
        return '-smp' not in self._start_parm.get('additional_switches', '')

    @property
    def post_processing(self):
        """Post-process an active MAPDL session.

        Examples
        --------
        Get the nodal displacement in the X direction for the first
        result set.

        >>> mapdl.set(1, 1)
        >>> disp_x = mapdl.post_processing.nodal_displacement('X')
        array([1.07512979e-04, 8.59137773e-05, 5.70690047e-05, ...,
               5.70333124e-05, 8.58600402e-05, 1.07445726e-04])
        """
        if self._exited:
            raise RuntimeError('MAPDL exited.\n\nCan only postprocess a live '
                               'MAPDL instance.')
        return self._post

    @property
    def chain_commands(self):
        """Chain several mapdl commands."""
        return self._chain_commands(self)

    def _chain_stored(self):
        """Send a series of commands to MAPDL"""
        # there's to be an limit to 640 characters per command, so
        # when chaining commands they must be shorter than 640 (minus
        # some overhead).
        c = 0
        chained_commands = []
        chunk = []
        for command in self._stored_commands:
            len_command = len(command) + 1  # include sep var
            if len_command + c > MAX_COMMAND_LENGTH:
                chained_commands.append('$'.join(chunk))
                chunk = [command]
                c = 0
            else:
                chunk.append(command)
                c += len_command

        # join the last
        chained_commands.append('$'.join(chunk))
        self._stored_commands = []

        responses = [self._run(command) for command in chained_commands]
        self._response = '\n'.join(responses)

    @property
    def parameters(self):
        """Collection of MAPDL parameters obtainable from the ``*GET``
        command or ``GET`` command.

        Examples
        --------
        Simply list all parameters except for MAPDL MATH parameters

        >>> mapdl.parameters
        ARR                              : ARRAY DIM (3, 1, 1)
        PARM_FLOAT                       : 20.0
        PARM_INT                         : 10.0
        PARM_LONG_STR                    : "stringstringstringstringstringst"
        PARM_STR                         : "string"
        PORT                             : 50052.0

        Get a parameter

        >>> mapdl.parameters['PARM_FLOAT']
        20.0

        Get an array parameter

        >>> mapdl.parameters['ARR']
        array([1., 2., 3.])

        Get the current routine

        >>> mapdl.parameters.routine
        'PREP7'

        >>> mapdl.parameters.units
        'SI'

        >>> mapdl.parameters.csys
        0
        """
        return self._parameters

    class _non_interactive:
        """Allows user to enter commands that need to run
        non-interactively.

        Examples
        --------
        To use an non-interactive command like *VWRITE, use:

        >>> with ansys.non_interactive:
                ansys.run("*VWRITE,LABEL(1),VALUE(1,1),VALUE(1,2),VALUE(1,3)")
                ansys.run("(1X,A8,'   ',F10.1,'  ',F10.1,'   ',1F5.3)")

        """
        def __init__(self, parent):
            self._parent = weakref.ref(parent)

        def __enter__(self):
            self._parent()._log.debug('Entering non-interactive mode')
            self._parent()._store_commands = True

        def __exit__(self, *args):
            self._parent()._log.debug('Entering non-interactive mode')
            self._parent()._flush_stored()

    class _chain_commands:
        """Store MAPDL commands and send one chained command."""

        def __init__(self, parent):
            self._parent = weakref.ref(parent)

        def __enter__(self):
            self._parent()._log.debug('Entering chained command mode')
            self._parent()._store_commands = True

        def __exit__(self, *args):
            self._parent()._log.debug('Entering chained command mode')
            self._parent()._chain_stored()
            self._parent()._store_commands = False

    @property
    def last_response(self):
        """Returns the last response from MAPDL.

        Examples
        --------
        >>> mapdl.last_response()
        'KEYPOINT      1   X,Y,Z=   1.00000       1.00000       1.00000'
        """
        return self._response

    def clear(self, *args, **kwargs):
        """Clear the database.

        APDL Command: ``/CLEAR``

        Examples
        --------
        >>> mapdl.clear()

        Notes
        -----
        Resets the ANSYS database to the conditions at the beginning
        of the problem.  Sets the import and Boolean options back to
        the ANSYS default. All items are deleted from the database and
        memory values are set to zero for items derived from database
        information.  All files are left intact.  This command is
        useful between multiple analyses in the same run, or between
        passes of a multi-pass analysis (such as between the
        substructure generation, use, and expansion passes).  Should
        not be used in a do-loop since loop counters will be reset.
        on the same line as the ``/CLEAR`` command.

        ``/CLEAR`` resets the jobname to match the currently open
        session .LOG and .ERR files. This will return the jobname to
        its original value, or to the most recent value specified on
        ``/FILNAME`` with KEY = 1.

        this command is valid only at the Begin level.

        """
        self.run('/CLE,NOSTART', mute=True)

    @supress_logging
    def __str__(self):
        try:
            if self._exited:
                return 'MAPDL exited'
            stats = self.slashstatus('PROD')
        except:  # pragma: no cover
            return 'MAPDL exited'

        st = stats.find('*** Products ***')
        en = stats.find('*** PrePro')
        product = '\n'.join(stats[st:en].splitlines()[1:]).strip()

        # get product version
        stats = self.slashstatus('TITLE')
        st = stats.find('RELEASE')
        en = stats.find('INITIAL', st)
        mapdl_version = stats[st:en].split('CUSTOMER')[0].strip()

        info =  'Product:         %s\n' % product
        info += 'MAPDL Version:   %s\n' % mapdl_version
        info += 'PyMAPDL Version: %s\n' % pymapdl.__version__
        return info

    @property
    def geometry(self):
        """Geometry (CAD) information."""
        return self._geometry

    @property
    def _geometry(self):  # pragma: no cover
        """Return geometry cache"""
        from ansys.mapdl.core.mapdl_geometry import Geometry
        return Geometry(self)

    @property
    def mesh(self):
        """Mesh information.

        Examples
        --------
        Return an array of the active nodes

        >>> mapdl.mesh.nodes
        array([[ 1.,  0.,  0.],
               [ 2.,  0.,  0.],
               [ 3.,  0.,  0.],
               [ 4.,  0.,  0.],
               [ 5.,  0.,  0.],
               [ 6.,  0.,  0.],
               [ 7.,  0.,  0.],
               [ 8.,  0.,  0.],
               [ 9.,  0.,  0.],
               [10.,  0.,  0.]])

        Return an array of the node numbers of the active nodes

        >>> mapdl.mesh.nnum
        array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10], dtype=int32)

        Simply query and print the geometry

        >>> print(mapdl.mesh)
          ANSYS Mapdl Mesh
          Number of Nodes:              321
          Number of Elements:           40
          Number of Element Types:      1
          Number of Node Components:    2
          Number of Element Components: 2

        Access the geometry as a VTK object

        >>> mapdl.mesh.grid

        """
        return self._mesh

    @property
    @supress_logging
    def _mesh(self):
        """Write entire archive to ASCII and read it in as an
        ``ansys.mapdl.core.Archive``"""
        # lazy import here to avoid loading pyvista and vtk
        from ansys.mapdl.reader import Archive

        if self._archive_cache is None:
            # write database to an archive file
            arch_filename = os.path.join(self.directory, '_tmp.cdb')
            nblock_filename = os.path.join(self.directory, 'nblock.cdb')

            # must have all nodes elements are using selected
            if hasattr(self, 'mute'):
                old_mute = self.mute
                self.mute = True

            with self.chain_commands:
                self.cm('__NODE__', 'NODE')
                self.nsle('S')
                self.cdwrite('db', arch_filename)
                self.cmsel('S', '__NODE__', 'NODE')

                self.cm('__ELEM__', 'ELEM')
                self.esel('NONE')
                self.cdwrite('db', nblock_filename)
                self.cmsel('S', '__ELEM__', 'ELEM')

            if hasattr(self, 'mute'):
                self.mute = old_mute

            self._archive_cache = Archive(arch_filename, parse_vtk=False,
                                          name='Mesh')
            grid = self._archive_cache._parse_vtk(additional_checking=True)
            self._archive_cache._grid = grid

            # rare bug
            if grid is not None:
                if grid.n_points != self._archive_cache.n_node:
                    self._archive_cache = Archive(arch_filename, parse_vtk=True,
                                                  name='Mesh')

            # overwrite nodes in archive
            nblock = Archive(nblock_filename, parse_vtk=False)
            self._archive_cache._nodes = nblock._nodes
            self._archive_cache._nnum = nblock._nnum
            self._archive_cache._node_coord = None

        return self._archive_cache

    def _reset_cache(self):
        """Reset cached items"""
        self._archive_cache = None

    @property
    def allow_ignore(self):
        """Invalid commands will be ignored rather than exceptions

        A command executed in the wrong processor will raise an
        exception when ``allow_ignore=False``.  This is the default
        behavior.

        Examples
        --------
        >>> mapdl.post1()
        >>> mapdl.k(1, 0, 0, 0)
        Exception:  K is not a recognized POST1 command, abbreviation, or macro.

        Ignore these messages by setting allow_ignore=True

        >>> mapdl.allow_ignore = True
        2020-06-08 21:39:58,094 [INFO] : K is not a
        recognized POST1 command, abbreviation, or macro.  This
        command will be ignored.

        *** WARNING *** CP = 0.372 TIME= 21:39:58
        K is not a recognized POST1 command, abbreviation, or macro.
        This command will be ignored.

        """
        return self._allow_ignore

    @allow_ignore.setter
    def allow_ignore(self, value):
        """Set allow ignore"""
        self._allow_ignore = bool(value)

    def open_apdl_log(self, filename, mode='w'):
        """Start writing all APDL commands to an ANSYS input file.

        Parameters
        ----------
        filename : str
            Filename of the log.
        """
        if self._apdl_log is not None:
            raise RuntimeError('APDL command logging already enabled')

        self._log.debug('Opening ANSYS log file at %s', filename)
        self._apdl_log = open(filename, mode=mode, buffering=1)  # line buffered
        if mode != 'w':
            self._apdl_log.write('! APDL script generated using ansys.mapdl.core %s\n' % pymapdl.__version__)

    @supress_logging
    @run_as_prep7
    def _generate_iges(self):
        """Save IGES geometry representation to disk"""
        filename = os.path.join(self.directory, '_tmp.iges')
        self.igesout(filename, att=1, mute=True)
        return filename

    def open_gui(self, include_result=True):  # pragma: no cover
        """Saves existing database and opens up the APDL GUI.

        Parameters
        ----------
        include_result : bool, optional
            Allow the result file to be post processed in the GUI.

        Examples
        --------
        >>> mapdl.open_gui()
        """
        # lazy load here to avoid circular import
        from ansys.mapdl.core.launcher import get_ansys_path

        if not self._local:
            raise RuntimeError('``open_gui`` can only be called from a local '
                               'MAPDL instance')

        # specify a path for the temporary database
        temp_dir = tempfile.gettempdir()
        save_path = os.path.join(temp_dir, f'ansys_{random_string(10)}')
        if os.path.isdir(save_path):
            rmtree(save_path)
        os.mkdir(save_path)

        name = 'tmp'
        tmp_database = os.path.join(save_path, '%s.db' % name)
        if os.path.isfile(tmp_database):
            os.remove(tmp_database)

        # cache result file, version, and routine before closing
        resultfile = self._result_file
        version = self.version
        prior_processor = self.parameters.routine

        # finish, save and exit the server
        self.finish(mute=True)
        self.save(tmp_database, mute=True)
        self.exit()

        # copy result file to temp directory
        if include_result and self._result_file is not None:
            if os.path.isfile(resultfile):
                tmp_resultfile = os.path.join(save_path, '%s.rst' % name)
                copyfile(resultfile, tmp_resultfile)

        # write temporary input file
        start_file = os.path.join(save_path, 'start%s.ans' % version)
        with open(start_file, 'w') as f:
            f.write('RESUME\n')

        # some versions of ANSYS just look for "start.ans" when starting
        other_start_file = os.path.join(save_path, 'start.ans')
        with open(other_start_file, 'w') as f:
            f.write('RESUME\n')

        # issue system command to run ansys in GUI mode
        cwd = os.getcwd()
        os.chdir(save_path)
        exec_file = self._start_parm.get('exec_file',
                                         get_ansys_path(allow_input=False))
        nproc = self._start_parm.get('nproc', 2)
        add_sw = self._start_parm.get('additional_switches', '')
        os.system(f'cd "{save_path}" && "{exec_file}" -g -j {name} -np {nproc} {add_sw}')
        os.chdir(cwd)
        # Consider removing this temporary directory

        # reattach to a new session and reload database
        self._launch(self._start_parm)
        self.resume(tmp_database, mute=True)
        if prior_processor is not None:
            if 'BEGIN' not in prior_processor.upper():
                self.run(f'/{prior_processor}', mute=True)

    def _launch(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError('Implemented by child class')

    def _close_apdl_log(self):
        """Closes the APDL log"""
        if self._apdl_log is not None:
            self._apdl_log.close()
        self._apdl_log = None

    def nplot(self, knum="", vtk=None, **kwargs):
        """APDL Command: NPLOT

        Displays nodes.

        Parameters
        ----------
        knum : bool, int, optional
            Node number key:

            - ``False`` : No node numbers on display (default).
            - ``True`` : Include node numbers on display.

        vtk : bool, optional
            Plot the currently selected nodes using ``pyvista``.
            Defaults to current ``use_vtk`` setting as set on the
            initialization of MAPDL.

        Examples
        --------
        Plot using VTK while showing labels and changing the background

        >>> mapdl.prep7()
        >>> mapdl.n(1, 0, 0, 0)
        >>> mapdl.n(11, 10, 0, 0)
        >>> mapdl.fill(1, 11, 9)
        >>> mapdl.nplot(knum=True, vtk=True, background='w', color='k',
                        show_bounds=True)

        Plot without using VTK

        >>> mapdl.prep7()
        >>> mapdl.n(1, 0, 0, 0)
        >>> mapdl.n(11, 10, 0, 0)
        >>> mapdl.fill(1, 11, 9)
        >>> mapdl.nplot(vtk=False)

        Notes
        -----
        Only selected nodes [NSEL] are displayed.  Elements need not
        be defined.
        """
        # lazy import here to avoid top level import
        import pyvista as pv

        if vtk is None:
            vtk = self._use_vtk

        if vtk:
            kwargs.setdefault('title', 'MAPDL Node Plot')
            if not self.mesh.n_node:
                warnings.warn('There are no nodes to plot.')
                return general_plotter([], [], [], **kwargs)

            labels = []
            if knum:
                # must eliminate duplicate points or labeling fails miserably.
                pcloud = pv.PolyData(self.mesh.nodes)
                pcloud['labels'] = self.mesh.nnum
                pcloud.clean(inplace=True)

                labels = [{'points': pcloud.points, 'labels': pcloud['labels']}]
            points = [{'points': self.mesh.nodes}]
            return general_plotter([], points, labels, **kwargs)

        # otherwise, use the built-in nplot
        if isinstance(knum, bool):
            knum = int(knum)

        self._enable_interactive_plotting()
        return super().nplot(knum, **kwargs)

    def vplot(self, nv1="", nv2="", ninc="", degen="", scale="",
              vtk=None, quality=4, show_area_numbering=False,
              show_line_numbering=False,
              color_areas=False, show_lines=True,
              **kwargs):
        """Plot the selected volumes.

        APDL Command: VPLOT

        Parameters
        ----------
        nv1, nv2, ninc
            Display volumes from NV1 to NV2 (defaults to NV1) in steps
            of NINC (defaults to 1).  If NV1 = ALL (default), NV2 and
            NINC are ignored and all selected volumes [VSEL] are
            displayed.  Ignored when ``vtk=True``.

        degen
            Degeneracy marker.  ``"blank"`` No degeneracy marker is
            used (default), or ``"DEGE"``.  A red star is placed on
            keypoints at degeneracies (see the Modeling and Meshing
            Guide).  Not available if /FACET,WIRE is set.  Ignored
            when ``vtk=True``.

        scale
            Scale factor for the size of the degeneracy-marker star.  The scale
            is the size in window space (-1 to 1 in both directions) (defaults
            to .075).  Ignored when ``vtk=True``.

        vtk : bool, optional
            Plot the currently selected volumes using ``pyvista``.  As
            this creates a temporary surface mesh, this may have a
            long execution time for large meshes.

        quality : int, optional
            quality of the mesh to display.  Varies between 1 (worst)
            to 10 (best).  Applicable when ``vtk=True``.

        show_numbering : bool, optional
            Display line and keypoint numbers when ``vtk=True``.

        Examples
        --------
        Plot while displaying area numbers

        >>> mapdl.vplot(show_area_numbering=True)
        """
        if vtk is None:
            vtk = self._use_vtk

        if vtk:
            cm_name = '__tmp_area2__'
            self.cm(cm_name, 'AREA')
            self.aslv('S')  # select areas attached to active volumes
            self.aplot(vtk=vtk, color_areas=color_areas, quality=quality,
                       show_area_numbering=show_area_numbering,
                       show_line_numbering=show_line_numbering,
                       show_lines=show_lines, **kwargs)
            self.cmsel('S', cm_name, 'AREA')
        else:
            self._enable_interactive_plotting()
            return super().vplot(nv1=nv1, nv2=nv2, ninc=ninc, **kwargs)

    def aplot(self, na1="", na2="", ninc="", degen="", scale="",
              vtk=None, quality=4, show_area_numbering=False,
              show_line_numbering=False, color_areas=False,
              show_lines=False, **kwargs):
        """APDL Command: APLOT

        Displays the selected areas from ``na1`` to ``na2`` in steps
        of ``ninc``.

        Parameters
        ----------
        na1 : int, optional
            Minimum area to display.

        na2 : int, optional
            Maximum area to display

        ninc : int, optional
            Increment between minimum and maximum area.

        degen, str, optional
            Degeneracy marker.  This option is ignored when ``vtk=True``.

        scale
            Scale factor for the size of the degeneracy-marker star.
            The scale is the size in window space (-1 to 1 in both
            directions) (defaults to .075).  This option is ignored
            when ``vtk=True``.

        vtk : bool, optional
            Plot the currently selected areas using ``pyvista``.  As
            this creates a temporary surface mesh, this may have a
            long execution time for large meshes.

        quality : int, optional
            Quality of the mesh to display.  Varies between 1 (worst)
            to 10 (best) when ``vtk=True``.

        show_area_numbering : bool, optional
            Display area numbers when ``vtk=True``.

        show_line_numbering : bool, optional
            Display line numbers when ``vtk=True``.

        color_areas : bool, optional
            Randomly color areas when ``True`` and ``vtk=True``.

        show_lines : bool, optional
            Plot lines and areas.  Change the thickness of the lines
            with ``line_width=``

        Examples
        --------
        Plot areas between 1 and 4 in increments of 2.

        >>> mapdl.block(0, 1, 0, 1, 0, 1)
        >>> mapdl.aplot(1, 4, 2)

        Plot all areas and randomly color the areas.  Label center of
        areas by their number.

        >>> mapdl.aplot(show_area_numbering=True, color_areas=True)


        """
        if vtk is None:
            vtk = self._use_vtk

        if vtk:
            kwargs.setdefault('title', 'MAPDL Area Plot')
            kwargs.setdefault('stitle', None)
            if quality > 10:
                quality = 10
            if quality < 1:
                quality = 1
            surf = self.geometry.generate_surface(11 - quality, na1, na2, ninc)
            meshes = []
            labels = []

            # individual surface isolation is quite slow, so just
            # color individual areas
            if color_areas:
                anum = surf['entity_num']
                rand = np.random.random(anum[-1] + 1)
                area_color = rand[anum]
                meshes.append({'mesh': surf, 'scalars': area_color})
            else:
                meshes.append({'mesh': surf})

            if show_area_numbering:
                anums = np.unique(surf['entity_num'])
                centers = []
                for anum in anums:
                    area = surf.extract_cells(surf['entity_num'] == anum)
                    centers.append(area.center)

                labels.append({'points': np.array(centers), 'labels': anums})

            if show_lines or show_line_numbering:
                kwargs.setdefault('line_width', 2)
                # subselect lines belonging to the current areas
                self.cm('__area__', 'AREA')
                self.lsla('S')

                lines = self.geometry.lines
                self.cmsel('S', '__area__', 'AREA')

                if show_lines:
                    meshes.append({'mesh': lines,
                                   'color': kwargs.get('edge_color', 'k')})
                if show_line_numbering:
                    labels.append({'points': lines.points[50::101],
                                   'labels': lines['entity_num']})

            return general_plotter(meshes, [],
                                   labels, **kwargs)

        else:
            self._enable_interactive_plotting()
            return super().aplot(na1=na1, na2=na2, ninc=ninc, **kwargs)

    @supress_logging
    def _enable_interactive_plotting(self, pixel_res=1600):
        """Enables interactive plotting.  Requires matplotlib

        Parameters
        ----------
        pixel_res : int
            Pixel resolution.  Valid values are from 256 to 2400.
            Lowering the pixel resolution produces a "fuzzier" image.
            Increasing the resolution produces a "sharper" image but
            takes longer to render.
        """
        if not self._has_matplotlib:
            raise ImportError('Install matplotlib to display plots from MAPDL ,'
                              'from Python.  Otherwise, plot with vtk with:\n'
                              '``vtk=True``')

        if not self._png_mode:
            self.show('PNG', mute=True)
            self.gfile(pixel_res, mute=True)

    @property
    def _has_matplotlib(self):
        try:
            import matplotlib
            return True
        except ImportError:
            return False

    @property
    def _png_mode(self):
        """Returns True when MAPDL is set to write plots as png to file."""
        return 'PNG' in self.show(mute=False)

    def set_log_level(self, loglevel):
        """Sets log level

        Parameters
        ----------
        loglevel : str, int
            Log level.  Must be one of: ``'DEBUG', 'INFO', 'WARNING', 'ERROR'``.

        Examples
        --------
        Set the log level to debug

        >>> mapdl.set_log_level('DEBUG')

        Set the log level to info

        >>> mapdl.set_log_level('INFO')

        Set the log level to warning

        >>> mapdl.set_log_level('WARNING')

        Set the log level to error

        >>> mapdl.set_log_level('ERROR')
        """
        if isinstance(loglevel, str):
            loglevel = loglevel.upper()
        setup_logger(loglevel=loglevel)

    def _list(self, command):
        """ Replaces *LIST command """
        items = command.split(',')
        filename = os.path.join(self.directory, '.'.join(items[1:]))
        if os.path.isfile(filename):
            self._response = open(filename).read()
            self._log.info(self._response)
        else:
            raise Exception('Cannot run:\n{command}\n\nFile does not exist')

    def eplot(self, show_node_numbering=False, vtk=None, **kwargs):
        """Plots the currently selected elements.

        APDL Command: EPLOT

        Parameters
        ----------
        vtk : bool, optional
            Plot the currently selected elements using ``pyvista``.
            Defaults to current ``use_vtk`` setting.

        show_node_numbering : bool, optional
            Plot the node numbers of surface nodes.

        **kwargs
            See ``help(ansys.mapdl.core.plotter.general_plotter)`` for more
            keyword arguments related to visualizing using ``vtk``.

        Examples
        --------
        >>> mapdl.clear()
        >>> mapdl.prep7()
        >>> mapdl.block(0, 1, 0, 1, 0, 1)
        >>> mapdl.et(1, 186)
        >>> mapdl.esize(0.1)
        >>> mapdl.vmesh('ALL')
        >>> mapdl.vgen(2, 'all')
        >>> mapdl.eplot(show_edges=True, smooth_shading=True,
                        show_node_numbering=True)

        Save a screenshot to disk without showing the plot

        >>> mapdl.eplot(background='w', show_edges=True, smooth_shading=True,
                        window_size=[1920, 1080], screenshot='screenshot.png', 
                        off_screen=True)

        """
        if vtk is None:
            vtk = self._use_vtk

        if vtk:
            kwargs.setdefault('title', 'MAPDL Element Plot')
            if not self._mesh.n_elem:
                warnings.warn('There are no elements to plot.')
                return general_plotter([], [], [], **kwargs)

            # TODO: Consider caching the surface
            esurf = self.mesh._grid.linear_copy().extract_surface().clean()
            kwargs.setdefault('show_edges', True)

            # if show_node_numbering:
            labels = []
            if show_node_numbering:
                labels = [{'points': esurf.points, 'labels': esurf['ansys_node_num']}]

            return general_plotter([{'mesh': esurf}],
                                   [],
                                   labels,
                                   **kwargs)

        # otherwise, use MAPDL plotter
        self._enable_interactive_plotting()
        return super().eplot(**kwargs)

    def lplot(self, nl1="", nl2="", ninc="", vtk=None,
              show_line_numbering=True,
              show_keypoint_numbering=False, color_lines=False,
              **kwargs):
        """APDL Command: LPLOT

        Displays the selected lines.

        Parameters
        ----------
        nl1, nl2, ninc
            Display lines from NL1 to NL2 (defaults to NL1) in steps
            of NINC (defaults to 1).  If NL1 = ALL (default), NL2 and
            NINC are ignored and display all selected lines [LSEL].

        vtk : bool, optional
            Plot the currently selected lines using ``pyvista``.

        show_line_numbering : bool, optional
            Display line and keypoint numbers when ``vtk=True``.

        show_keypoint_numbering : bool, optional
            Number keypoints.  Only valid when show_keypoints is True

        **kwargs
            See ``help(pyvista.plot)`` for more keyword arguments
            related to visualizing using ``vtk``.

        Examples
        --------
        >>> mapdl.lplot(vtk=True, cpos='xy', line_width=10)

        Notes
        -----
        Mesh divisions on plotted lines are controlled by the LDIV
        option of the /PSYMB command.

        This command is valid in any processor.
        """
        if vtk is None:
            vtk = self._use_vtk

        if vtk:
            kwargs.setdefault('title', 'MAPDL Line Plot')
            if not self.geometry.n_line:
                warnings.warn('Either no lines have been selected or there '
                              'is nothing to plot')
                return general_plotter([], [], [], **kwargs)

            lines = self.geometry.lines
            meshes = [{'mesh': lines}]
            if color_lines:
                meshes[0]['scalars'] = np.random.random(lines.n_cells)

            labels = []
            if show_line_numbering:
                labels.append({'points': lines.points[50::101],
                               'labels': lines['entity_num']})

            if show_keypoint_numbering:
                labels.append({'points': self.geometry.keypoints,
                               'labels': self.geometry.knum})

            return general_plotter(meshes,
                                   [],
                                   labels,
                                   **kwargs)
        else:
            self._enable_interactive_plotting()
            return super().lplot(nl1=nl1, nl2=nl2, ninc=ninc, **kwargs)

    def kplot(self, np1="", np2="", ninc="", lab="", vtk=None,
              show_keypoint_numbering=True, **kwargs):
        """Displays the selected keypoints.

        APDL Command: KPLOT

        Parameters
        ----------
        np1, np2, ninc
            Display keypoints from NP1 to NP2 (defaults to NP1) in
            steps of NINC (defaults to 1).  If NP1 = ALL (default),
            NP2 and NINC are ignored and all selected keypoints [KSEL]
            are displayed.

        lab
            Determines what keypoints are plotted (one of the following):

            (blank) - Plots all keypoints.

            HPT - Plots only those keypoints that are hard points.

        vtk : bool, optional
            Plot the currently selected lines using ``pyvista``.

        show_keypoint_numbering : bool, optional
            Display keypoint numbers when ``vtk=True``.

        Notes
        -----
        This command is valid in any processor.
        """
        if vtk is None:
            vtk = self._use_vtk

        if vtk:
            kwargs.setdefault('title', 'MAPDL Keypoint Plot')
            if not self.geometry.n_keypoint:
                warnings.warn('Either no keypoints have been '
                              'selected or there are no keypoints in '
                              'the database.')
                return general_plotter([], [], [], **kwargs)

            keypoints = self.geometry.keypoints
            points = [{'points': keypoints}]

            labels = []
            if show_keypoint_numbering:
                labels.append({'points': keypoints,
                               'labels': self.geometry.knum})

            return general_plotter([], points,
                                   labels, **kwargs)

        # otherwise, use the legacy plotter
        self._enable_interactive_plotting()
        return super().kplot(np1=np1, np2=np2, ninc=ninc, lab=lab, **kwargs)

    @property
    def result(self):
        """Binary interface to the result file using ``ansys.mapdl.reader.Result``

        Examples
        --------
        >>> mapdl.solve()
        >>> mapdl.finish()
        >>> result = mapdl.result
        >>> print(result)
        PyMAPDL-Reader Result file object
        Units       : User Defined
        Version     : 18.2
        Cyclic      : False
        Result Sets : 1
        Nodes       : 3083
        Elements    : 977

        Available Results:
        EMS : Miscellaneous summable items (normally includes face pressures)
        ENF : Nodal forces
        ENS : Nodal stresses
        ENG : Element energies and volume
        EEL : Nodal elastic strains
        ETH : Nodal thermal strains (includes swelling strains)
        EUL : Element euler angles
        EMN : Miscellaneous nonsummable items
        EPT : Nodal temperatures
        NSL : Nodal displacements
        RF  : Nodal reaction forces
        """
        from ansys.mapdl.reader import read_binary
        from ansys.mapdl.reader.rst import Result

        if not self._local:
            # download to temporary directory
            save_path = os.path.join(tempfile.gettempdir(), 'ansys_tmp')
            result_path = self.download_result(save_path)
        else:
            if self._distributed_result_file and self._result_file:
                result_path = self._distributed_result_file
                result = Result(result_path, read_mesh=False)
                if result._is_cyclic:
                    result_path = self._result_file
                else:
                    # return the file with the last access time
                    filenames = [self._distributed_result_file, self._result_file]
                    result_path = last_created(filenames)
                    if result_path is None:  # if same return result_file
                        result_path = self._result_file

            elif self._distributed_result_file:
                result_path = self._distributed_result_file
                result = Result(result_path, read_mesh=False)
                if result._is_cyclic:
                    if not os.path.isfile(self._result_file):
                        raise RuntimeError('Distributed Cyclic result not supported')
                    result_path = self._result_file
            else:
                result_path = self._result_file

        if result_path is None:
            raise FileNotFoundError('No result file(s) at %s' % self.directory)
        if not os.path.isfile(result_path):
            raise FileNotFoundError('No results found at %s' % result_path)

        return read_binary(result_path)

    @property
    def _result_file(self):
        """Path of the non-distributed result file"""
        try:
            filename = self.inquire('RSTFILE')
            if not filename:
                filename = self.jobname
        except:
            filename = self.jobname

        try:
            ext = self.inquire('RSTEXT')
        except:  # check if rth file exists
            ext = ''

        if ext == '':
            rth_file = os.path.join(self.directory, '%s.%s' % (filename, 'rth'))
            rst_file = os.path.join(self.directory, '%s.%s' % (filename, 'rst'))

            if os.path.isfile(rth_file) and os.path.isfile(rst_file):
                return last_created([rth_file, rst_file])
            elif os.path.isfile(rth_file):
                return rth_file
            elif os.path.isfile(rst_file):
                return rst_file
        else:
            filename = os.path.join(self.directory, '%s.%s' % (filename, ext))
            if os.path.isfile(filename):
                return filename

    @property
    def _distributed_result_file(self):
        """Path of the distributed result file """
        try:
            filename = self.inquire('RSTFILE')
            if not filename:
                filename = self.jobname
        except:
            filename = self.jobname

        # ansys decided that a jobname ended in a number needs a bonus "_"
        if filename[-1].isnumeric():
            filename += '_'

        rth_basename = '%s0.%s' % (filename, 'rth')
        rst_basename = '%s0.%s' % (filename, 'rst')

        rth_file = os.path.join(self.directory, rth_basename)
        rst_file = os.path.join(self.directory, rst_basename)
        if os.path.isfile(rth_file) and os.path.isfile(rst_file):
            return last_created([rth_file, rst_file])
        elif os.path.isfile(rth_file):
            return rth_file
        elif os.path.isfile(rst_file):
            return rst_file

    def _get(self, *args, **kwargs):
        """Simply use the default get method"""
        return self.get(*args, **kwargs)

    def add_file_handler(self, filepath, append=False, level='DEBUG'):
        """Add a file handler to the mapdl log.  This allows you to
        redirect the APDL logging to a file.

        Parameters
        ----------
        filepath : str
            Filename of the log.

        append : bool
            When ``True``, appends to an existing log file.  When
            ``False``, overwrites the log file if it already exists.

        level : str
            Log level.  Must be one of: ``'DEBUG', 'INFO', 'WARNING', 'ERROR'``.

        Examples
        --------
        Start writing the log to a new file named "mapdl.log"

        >>> mapdl.add_file_handler('mapdl.log')

        """
        if append:
            mode = 'a'
        else:
            mode = 'w'

        self._log_filehandler = logging.FileHandler(filepath)
        formatstr = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'

        self._log_filehandler = logging.FileHandler(filepath, mode=mode)
        self._log_filehandler.setFormatter(logging.Formatter(formatstr))
        if isinstance(level, str):
            level = level.uppder()
        self._log_filehandler.setLevel(level)
        self._log.addHandler(self._log_filehandler)
        self._log.info('Added file handler at %s', filepath)

    def remove_file_handler(self):
        """Removes the filehander from the log"""
        self._log.removeHandler(self._log_filehandler)
        self._log.info('Removed file handler')

    def _flush_stored(self):
        """Writes stored commands to an input file and runs the input
        file.  Used with non_interactive.
        """
        self._log.debug('Flushing stored commands')
        rnd_str = random_string()
        tmp_out = os.path.join(tempfile.gettempdir(), 'tmp_%s.out' % rnd_str)
        self._stored_commands.insert(0, "/OUTPUT, '%s'" % tmp_out)
        self._stored_commands.append('/OUTPUT')
        commands = '\n'.join(self._stored_commands)
        if self._apdl_log:
            self._apdl_log.write(commands + '\n')

        # write to a temporary input file
        tmp_inp = os.path.join(tempfile.gettempdir(), 'tmp_%s.inp' % rnd_str)
        self._log.debug('Writing the following commands to a temporary '
                        'apdl input file:\n%s', commands)

        with open(tmp_inp, 'w') as f:
            f.writelines(commands)

        self._store_commands = False
        self._stored_commands = []

        # interactive result
        _ = self.input(tmp_inp, write_to_log=False)
        time.sleep(0.1)  # allow MAPDL to close the file
        if os.path.isfile(tmp_out):
            self._response = '\n' + open(tmp_out).read()

        if self._response is None:
            self._log.warning('Unable to read response from flushed commands')
        else:
            self._log.info(self._response)

    def get_float(self, *args, **kwargs):
        raise NotImplementedError('Please use ``get_value`` instead')

    def get_value(self, entity="", entnum="", item1="", it1num="",
                  item2="", it2num="", **kwargs):
        """Runs the \*GET command and returns a Python value.

        See ``help(mapdl.starget)`` for more details.

        Parameters
        ----------
        entity
            Entity keyword. Valid keywords are NODE, ELEM, KP, LINE, AREA,
            VOLU, PDS, etc., as shown for Entity = in the tables below.

        entnum
            The number or label for the entity (as shown for ENTNUM = in the
            tables below). In some cases, a zero (or blank) ENTNUM represents
            all entities of the set.

        item1
            The name of a particular item for the given entity. Valid items are
            as shown in the Item1 columns of the tables below.

        it1num
            The number (or label) for the specified Item1 (if any). Valid
            IT1NUM values are as shown in the IT1NUM columns of the tables
            below. Some Item1 labels do not require an IT1NUM value.

        item2, it2num
            A second set of item labels and numbers to further qualify the item
            for which data are to be retrieved. Most items do not require this
            level of information.

        Returns
        -------
        par : float
            Floating point value of the parameter.

        Examples
        --------
        Retrieve the number of nodes.

        >>> value = ansys.get_value('node', '', 'count')
        >>> value
        3003

        Retrieve the number of nodes using keywords.

        >>> value = ansys.get_value(entity='node', item1='count')
        >>> value
        3003
        """
        return self._get(entity=entity, entnum=entnum, item1=item1, it1num=it1num,
                         item2=item2, it2num=it2num, **kwargs)

    def get(self, par="__floatparameter__", entity="", entnum="",
            item1="", it1num="", item2="", it2num="", **kwargs):
        """APDL Command: \*GET

        Retrieves a value and stores it as a scalar parameter or part
        of an array parameter.

        Parameters
        ----------
        par : str, optional
            The name of the resulting parameter. See \*SET for name
            restrictions.

        entity
            Entity keyword. Valid keywords are NODE, ELEM, KP, LINE,
            AREA, VOLU, PDS, etc., as shown for Entity = in the tables
            below.

        entnum
            The number or label for the entity (as shown for ENTNUM =
            in the tables below). In some cases, a zero (or blank)
            ENTNUM represents all entities of the set.

        item1
            The name of a particular item for the given entity.

        it1num
            The number (or label) for the specified Item1 (if
            any). Valid IT1NUM values are as shown in the IT1NUM
            columns of the tables below. Some Item1 labels do not
            require an IT1NUM value.

        item2, it2num
            A second set of item labels and numbers to further qualify
            the item for which data are to be retrieved. Most items do
            not require this level of information.

        Returns
        -------
        par : float
            Floating point value of the parameter.

        Examples
        --------
        Retrieve the number of nodes

        >>> value = mapdl.get('val', 'node', '', 'count')
        >>> value
        3003

        Retrieve the number of nodes using keywords.  Note that the
        parameter name is optional.

        >>> value = mapdl.get(entity='node', item1='count')
        >>> value
        3003

        Notes
        -----
        GET retrieves a value for a specified item and stores the
        value as a scalar parameter, or as a value in a user-named
        array parameter. An item is identified by various keyword,
        label, and number combinations.  Usage is similar to the SET
        command except that the parameter values are retrieved from
        previously input or calculated results. For example,
        GET,A,ELEM,5,CENT,X returns the centroid x-location of element
        5 and stores the result as parameter A. GET command
        operations, along with the associated Get functions return
        values in the active coordinate system unless stated
        otherwise. A Get function is an alternative in- line function
        that can be used to retrieve a value instead of the GET
        command (see Using In-line Get Functions for more
        information).

        Both GET and VGET retrieve information from the active data
        stored in memory. The database is often the source, and
        sometimes the information is retrieved from common memory
        blocks that the program uses to manipulate
        information. Although POST1 and POST26 operations use a .rst
        file, GET data is accessed from the database or from the
        common blocks. Get operations do not access the .rst file
        directly. For repeated gets of sequential items, such as from
        a series of elements, see the VGET command.

        Most items are stored in the database after they are
        calculated and are available anytime thereafter. Items are
        grouped according to where they are usually first defined or
        calculated. Preprocessing data will often not reflect the
        calculated values generated from section data. Do not use GET
        to obtain data from elements that use calculated section data,
        such as beams or shells. Most of the general items listed
        below are available from all modules.

        """
        command = f'*GET,{par},{entity},{entnum},{item1},{it1num},{item2},{it2num}'
        kwargs['mute'] = False
        response = self.run(command, **kwargs)
        value = response.split('=')[-1].strip()
        try:  # always either a float or string
            return float(value)
        except:
            return value

    @property
    def jobname(self):
        """MAPDL job name.

        This is requested from the active mapdl instance.
        """
        try:
            self._jobname = self.inquire('JOBNAME')
        except:
            pass
        return self._jobname

    @jobname.setter
    def jobname(self, new_jobname):
        """Set the jobname"""
        self.finish(mute=True)
        self.filname(new_jobname, mute=True)
        self._jobname = new_jobname

    @supress_logging
    def inquire(self, func):
        """Returns system information.

        Parameters
        ----------
        func : str
           Specifies the type of system information returned.  See the
           notes section for more information.

        Returns
        -------
        value : str
            Value of the inquired item.

        Notes
        -----
        Allowable func entries
        LOGIN - Returns the pathname of the login directory on Linux
        systems or the pathname of the default directory (including
        drive letter) on Windows systems.

        - ``DOCU`` - Pathname of the ANSYS docu directory.
        - ``APDL`` - Pathname of the ANSYS APDL directory.
        - ``PROG`` - Pathname of the ANSYS executable directory.
        - ``AUTH`` - Pathname of the directory in which the license file resides.
        - ``USER`` - Name of the user currently logged-in.
        - ``DIRECTORY`` - Pathname of the current directory.
        - ``JOBNAME`` - Current Jobname.
        - ``RSTDIR`` - Result file directory
        - ``RSTFILE`` - Result file name
        - ``RSTEXT`` - Result file extension
        - ``OUTPUT`` - Current output file name

        Examples
        --------
        Return the job name

        >>> mapdl.inquire('JOBNAME')
        file

        Return the result file name

        >>> mapdl.inquire('RSTFILE')
        'file.rst'
        """
        response = self.run(f'/INQUIRE,,{func}', mute=False)
        return response.split('=')[1].strip()

    def modal_analysis(self, method='lanb', nmode='', freqb='', freqe='', cpxmod='',
                       nrmkey='', modtype='', memory_option='', elcalc=False):
        """Run a modal with basic settings analysis

        Parameters
        ----------
        method : str
            Mode-extraction method to be used for the modal analysis.
            Defaults to lanb (block lanczos).  Must be one of the following:

            - LANB : Block Lanczos
            - LANPCG : PCG Lanczos
            - SNODE : Supernode modal solver
            - SUBSP : Subspace algorithm
            - UNSYM : Unsymmetric matrix
            - DAMP : Damped system
            - QRDAMP : Damped system using QR algorithm
            - VT : Variational Technology

        nmode : int, optional
            The number of modes to extract. The value can depend on
            the value supplied for Method. NMODE has no default and
            must be specified. If Method = LANB, LANPCG, or SNODE, the
            number of modes that can be extracted can equal the DOFs
            in the model after the application of all boundary
            conditions.

        freqb : float, optional
            The beginning, or lower end, of the frequency range of
            interest.

        freqe : float, optional
            The ending, or upper end, of the frequency range of
            interest (in Hz). The default for Method = SNODE is
            described below. The default for all other methods is to
            calculate all modes, regardless of their maximum
            frequency.

        cpxmod : str, optional
            Complex eigenmode key. Valid only when ``method='QRDAMP'``
            or ``method='unsym'``

            - AUTO : Determine automatically if the eigensolutions are
              real or complex and output them accordingly. This is
              the default for ``method='UNSYM'``.  Not supported for
              Method = QRDAMP.
            - ON or CPLX : Calculate and output complex eigenmode
              shapes.
            - OFF or REAL : Do not calculate complex eigenmode
              shapes. This is required if a mode-
              superposition analysis is intended after the
              modal analysis for Method = QRDAMP. This is the
              default for this method.

        nrmkey : bool, optional
            Mode shape normalization key.  When ``True`` (default),
            normalize the mode shapes to the mass matrix.  When False,
            Normalize the mode shapes to unity instead of to the mass
            matrix.  If a subsequent spectrum or mode-superposition
            analysis is planned, the mode shapes should be normalized
            to the mass matrix.

        modtype : str, optional
            Type of modes calculated by the eigensolver. Only
            applicable to the unsymmetric eigensolver.

            - Blank : Right eigenmodes. This value is the default.
            - BOTH : Right and left eigenmodes. The left eigenmodes are
              written to Jobname.LMODE.  This option must be
              activated if a mode-superposition analysis is intended.

        memory_option : str, optional
            Memory allocation option:

            * ``DEFAULT`` - Default Memory mode
                      Use the default memory allocation strategy for
                      the sparse solver. The default strategy attempts
                      to run in the INCORE memory mode. If there is
                      not enough available physical memory when the
                      solver starts to run in the ``INCORE`` memory
                      mode, the solver will then attempt to run in the
                      ``OUTOFCORE`` memory mode.

            * ``INCORE`` - In-core memory mode
                     Use a memory allocation strategy in the sparse
                     solver that will attempt to obtain enough memory
                     to run with the entire factorized matrix in
                     memory. This option uses the most amount of
                     memory and should avoid doing any I/O. By
                     avoiding I/O, this option achieves optimal solver
                     performance. However, a significant amount of
                     memory is required to run in this mode, and it is
                     only recommended on machines with a large amount
                     of memory. If the allocation for in-core memory
                     fails, the solver will automatically revert to
                     out-of-core memory mode.

            * ``OUTOFCORE`` - Out of core memory mode.
                        Use a memory allocation strategy in the sparse
                        solver that will attempt to allocate only
                        enough work space to factor each individual
                        frontal matrix in memory, but will store the
                        entire factorized matrix on disk. Typically,
                        this memory mode results in poor performance
                        due to the potential bottleneck caused by the
                        I/O to the various files written by the
                        solver.

        elcalc : bool, optional
            Calculate element results, reaction forces, energies, and
            the nodal degree of freedom solution.  Default ``False``.

        Returns
        -------
        response : str
            Output from MAPDL SOLVE command.

        Examples
        --------
        Modal analysis using default parameters for the first 6 modes

        >>> mapdl.modal_analysis(nmode=6)

        Notes
        -----
        For models that involve a non-symmetric element stiffness
        matrix, as in the case of a contact element with frictional
        contact, the QRDAMP eigensolver (MODOPT, QRDAMP) extracts
        modes in the modal subspace formed by the eigenmodes from the
        symmetrized eigenproblem. The QRDAMP eigensolver symmetrizes
        the element stiffness matrix on the first pass of the
        eigensolution, and in the second pass, eigenmodes are
        extracted in the modal subspace of the first eigensolution
        pass. For such non- symmetric eigenproblems, you should verify
        the eigenvalue and eigenmode results using the non-symmetric
        matrix eigensolver (MODOPT,UNSYM).

        The DAMP and QRDAMP options cannot be followed by a subsequent
        spectrum analysis. The UNSYM method supports spectrum analysis
        when eigensolutions are real.

        """
        if nrmkey:
            if nrmkey.upper() != 'OFF':
                nrmkey = 'ON'
        nrmkey = 'OFF'

        with self.chain_commands:
            self.slashsolu()
            self.antype(2, 'new')
            self.modopt(method, nmode, freqb, freqe, cpxmod, nrmkey, modtype)
            self.bcsoption(memory_option)

            if elcalc:
                self.mxpand(elcalc='YES')

        out = self.solve()
        self.finish(mute=True)
        return out

    def run_multiline(self, commands):
        """Run several commands as a single block

        Parameters
        ----------
        commands : str
            Commands separated by new lines.  See example.

        Returns
        -------
        str
            Command output from MAPDL.  Includes the output from
            running every command, as if it was an input file.

        Examples
        --------
        Run several commands from Python multi-line string.

        >>> cmd = '''/prep7
        ! Mat
        MP,EX,1,200000
        MP,NUXY,1,0.3
        MP,DENS,1,7.85e-09
        ! Elements
        et,1,186
        et,2,154
        ! Geometry
        BLC4,0,0,1000,100,10
        ! Mesh
        esize,5
        vmesh,all
        nsel,s,loc,x,0
        d,all,all
        nsel,s,loc,x,999,1001
        type,2
        esurf
        esel,s,type,,2
        nsle
        sfe,all,3,pres,,-10
        allsel
        /solu
        antype,0
        solve
        /post1
        set,last
        plnsol,u,sum
        '''
        >>> resp = mapdl.run_multiline(cmd)
        >>> resp
        MATERIAL          1     EX   =   200000.0
        MATERIAL          1     NUXY =  0.3000000
        MATERIAL          1     DENS =  0.7850000E-08
        ELEMENT TYPE          1 IS SOLID186     3-D 20-NODE STRUCTURAL SOLID
         KEYOPT( 1- 6)=        0      0      0        0      0      0
         KEYOPT( 7-12)=        0      0      0        0      0      0
         KEYOPT(13-18)=        0      0      0        0      0      0
        output continues...

        """
        self._stored_commands = commands.splitlines()
        self._flush_stored()
        return self._response

    def run(self, command, write_to_log=True, **kwargs):
        """Run single APDL command.

        For multiple commands, use ``run_multiline``.

        Parameters
        ----------
        command : str
            ANSYS APDL command.

        write_to_log : bool, optional
            Overrides APDL log writing.  Default ``True``.  When set
            to ``False``, will not write command to log, even if APDL
            command logging is enabled.

        kwargs : Optional keyword arguments
            These keyword arguments are interface specific.

        Returns
        -------
        command_output : str
            Command output from MAPDL.

        Examples
        --------
        >>> mapdl.run('/PREP7')

        Equivalent Pythonic method:

        >>> mapdl.prep7()

        Notes
        -----
        When two or more commands need to be run non-interactively
        (i.e. ``*VWRITE``) use

        >>> with mapdl.non_interactive:
        ...     mapdl.run("*VWRITE,LABEL(1),VALUE(1,1),VALUE(1,2),VALUE(1,3)")
        ...     mapdl.run("(1X,A8,'   ',F10.1,'  ',F10.1,'   ',1F5.3)")

        Alternatively, you can simply run a block of commands with:

        >>> mapdl.run_multiline(cmd)
        """
        command = command.strip()
        # check if multiline
        if '\n' in command or '\r' in command:
            raise ValueError('Use ``run_multiline`` for multi-line commands')

        # always reset the cache
        self._reset_cache()

        # address MAPDL /INPUT level issue
        if command[:4].upper() == '/CLE':
            # Address gRPC issue
            # https://github.com/pyansys/pymapdl/issues/380
            command = '/CLE,NOSTART'

        if self._store_commands:
            self._stored_commands.append(command)
            return
        elif command[:3].upper() in INVAL_COMMANDS:
            exception = RuntimeError('Invalid pymapdl command "%s"\n\n%s' %
                                     (command, INVAL_COMMANDS[command[:3]]))
            raise exception
        elif command[:4].upper() in INVAL_COMMANDS:
            exception = RuntimeError('Invalid pymapdl command "%s"\n\n%s' %
                                     (command, INVAL_COMMANDS[command[:4]]))
            raise exception
        elif write_to_log and self._apdl_log is not None:
            if not self._apdl_log.closed:
                self._apdl_log.write('%s\n' % command)

        if command[:4].upper() == '/LIS':
            # simply return the contents of the file
            return self.list(*command.split(',')[1:])

        text = self._run(command, **kwargs)
        text = text.replace('\\r\\n', '\n').replace('\\n', '\n')
        if text:
            self._response = text.strip()
            self._log.info(self._response)
        else:
            self._response = ''
            return self._response

        if 'is not a recognized' in text:
            if not self.allow_ignore:
                text = text.replace('This command will be ignored.', '')
                text += '\n\nIgnore these messages by setting allow_ignore=True'
                raise MapdlInvalidRoutineError(text)

        # flag errors
        if '*** ERROR ***' in self._response and not self._ignore_errors:
            # remove permitted errors and allow MAPDL to continue
            response = self._response
            for err_str in _PERMITTED_ERRORS:
                response = re.sub(err_str, '', response)

            if '*** ERROR ***' in response:
                self._log.error(self._response)
                raise MapdlRuntimeError(self._response)
            else:
                warnings.warn('MAPDL returned non-abort errors.  Please '
                              'check the logs.')

        # special returns for certain geometry commands
        short_cmd = parse_to_short_cmd(command)

        # command parsing
        if short_cmd in element_commands:
            return element_commands[short_cmd](self._response)
        if short_cmd in PLOT_COMMANDS:
            return self._display_plot(self._response)

        return self._response

    @property
    def ignore_errors(self):
        """Flag to ignore MAPDL errors.

        Normally, any string containing "*** ERROR ***" from MAPDL
        will trigger a ``MapdlRuntimeError``.  Set this to ``True`` to
        ignore these errors.
        """
        return self._ignore_errors

    @ignore_errors.setter
    def ignore_errors(self, value):
        self._ignore_errors = bool(value)

    def load_table(self, name, array, var1='', var2='', var3=''):
        """Load a table from Python to MAPDL.

        Uses TREAD to transfer the table.

        Parameters
        ----------
        name : str
            An alphanumeric name used to identify this table.  Name
            may be up to 32 characters, beginning with a letter and
            containing only letters, numbers, and underscores.
            Examples: ``"ABC" "A3X" "TOP_END"``.

        array : np.ndarray or List
            List as a table or ``numpy`` array.

        var1 : str, optional
            A primary variable (listed below) or can be an independent
            parameter. If specifying an independent parameter, then
            you must define an additional table for the independent
            parameter. The additional table must have the same name as
            the independent parameter and may be a function of one or
            more primary variables or another independent
            parameter. All independent parameters must relate to a
            primary variable.

            - ``"TIME"``: Time
            - ``"FREQ"``: Frequency
            - ``"X"``: X-coordinate location
            - ``"Y"``: Y-coordinate location
            - ``"Z"``: Z-coordinate location
            - ``"TEMP"``: Temperature
            - ``"VELOCITY"``: Velocity
            - ``"1"``]: Pressure	PRESSURE [
            - ``"GAP"``: Geometric gap/penetration
            - ``"SECTOR"``: Cyclic sector number
            - ``"OMEGS"``: Amplitude of the rotational velocity vector
            - ``"ECCENT"``: Eccentricity
            - ``"THETA"``: Phase shift
            - ``"ELEM"``: Element number
            - ``"NODE"``: Node number
            - ``"CONC"``: Concentration

        var2 : str, optional
            See ``var1``

        var3 : str, optional
            See ``var1``

        Examples
        --------
        >>> my_conv = np.array([[0, 0.001],
                                [120, 0.001],
                                [130, 0.005],
                                [700, 0.005],
                                [710, 0.002],
                                [1000, 0.002]])
        >>> mapdl.load_table('MY_TABLE', my_conv, 'TIME')
        >>> mapdl.parameters['MY_TABLE']
        array([0.0001, 0.0001, 0.0005, 0.0005, 0.0002, 0.0002])
        """
        if array.ndim < 2:
            raise ValueError('Expecting at least a 2D table, but input contains '
                             'only 1 dimension')
        self.dim(name, 'TABLE', array.shape[0], var1=var1, var2=var2, var3=var3)
        base_name = random_string() + '.txt'
        filename = os.path.join(tempfile.gettempdir(), base_name)
        np.savetxt(filename, array)

        if not self._local:
            self.upload(filename, progress_bar=False)
            filename = base_name
        self.tread(name, filename, mute=True)

    def _display_plot(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError('Implemented by child class')

    def _run(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError('Implemented by child class')

    @property
    def version(self):
        """MAPDL build version

        Examples
        --------
        >>> mapdl.version
        20.2
        """
        return self.parameters.revision

    @property
    @supress_logging
    def directory(self):
        """Current MAPDL directory

        Examples
        --------
        Directory on Linux

        >>> mapdl.directory
        '/tmp/ansys'

        Directory on Windows

        >>> mapdl.directory
        'C:/temp_directory/'

        """
        # always attempt to cache the path
        try:
            self._path = self.inquire('DIRECTORY')
        except:
            pass

        # os independent path format
        if self._path is not None:
            self._path = self._path.replace('\\', '/')
            # new line to fix path issue, see #416
            self._path = repr(self._path)[1:-1]
        return self._path

    @property
    def _lockfile(self):
        """lockfile path"""
        path = self.directory
        if path is not None:
            return os.path.join(path, self.jobname + '.lock').replace('\\', '/')

    def exit(self):  # pragma: no cover
        """Exit from MAPDL"""
        raise NotImplementedError('Implemented by child class')

    def __del__(self):  # pragma: no cover
        """Clean up when complete"""
        if self._cleanup:
            try:
                self.exit()
            except Exception as e:
                try:  # logger might be closed
                    if self._log is not None:
                        self._log.error('exit: %s', str(e))
                except:
                    pass

    @supress_logging
    def get_array(self, entity='', entnum='', item1='', it1num='', item2='',
                  it2num='', kloop='', **kwargs):
        """Uses the VGET command to Return an array from ANSYS as a
        Python array.

        Parameters
        ----------
        entity
            Entity keyword.  Valid keywords are NODE, ELEM, KP, LINE,
            AREA, VOLU, etc

        entnum
            The number of the entity (as shown for ENTNUM = in the tables
            below).

        item1
            The name of a particular item for the given entity.  Valid
            items are as shown in the Item1 columns of the tables
            below.

        it1num
            The number (or label) for the specified Item1 (if any).
            Valid IT1NUM values are as shown in the IT1NUM columns of
            the tables below.  Some Item1 labels do not require an
            IT1NUM value.

        item2, it2num
            A second set of item labels and numbers to further qualify
            the item for which data is to be retrieved.  Most items do
            not require this level of information.

        kloop
            Field to be looped on:

            - 0 or 2 : Loop on the ENTNUM field (default).
            - 3 : Loop on the Item1 field.
            - 4 : Loop on the IT1NUM field. Successive items are as shown with IT1NUM.
            - 5 : Loop on the Item2 field.
            - 6 : Loop on the IT2NUM field. Successive items are as shown with IT2NUM.

        Examples
        --------
        List the current selected node numbers

        >>> mapdl.get_array('NODE', item1='NLIST')
        array([  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,
              ...
              314., 315., 316., 317., 318., 319., 320., 321.])

        List the displacement in the X direction for the first result

        >>> mapdl.post1()
        >>> mapdl.set(1, 1)
        >>> disp_x = mapdl.get_array('NODE', item1='U', it1num='X')
        array([ 0.01605306, -0.01605306,  0.00178402, -0.01605306,
               ...
               -0.00178402, -0.01234851,  0.01234851, -0.01234851])

        Notes
        -----
        Please reference your ANSYS help manual *VGET command tables
        for all the available *VGET values
        """
        arr = self._get_array(entity, entnum, item1, it1num, item2,
                              it2num, kloop)

        # edge case where corba refuses to return the array
        ntry = 0
        while arr.size == 1 and arr[0] == -1:
            arr = self._get_array(entity, entnum, item1, it1num,
                                  item2, it2num, kloop)
            if ntry > 5:
                raise RuntimeError('Unable to get array for %s' % entity)
        return arr

    def _get_array(self, entity='', entnum='', item1='', it1num='', item2='',
                   it2num='', kloop='', dtype=None, **kwargs):
        """Uses the VGET command to get an array from ANSYS"""
        parm_name = kwargs.pop('parm', None)

        if parm_name is None:
            parm_name = '__vget_tmp_%d__' % self._vget_arr_counter
            self._vget_arr_counter += 1

        out = self.starvget(parm_name, entity, entnum, item1, it1num, item2,
                            it2num, kloop, mute=False)

        # check if empty array
        if 'the dimension number 1 is 0' in out:
            return np.empty(0)

        with self.non_interactive:
            self.vwrite('%s(1)' % parm_name)
            self.run('(F20.12)')

        array = np.fromstring(self.last_response, sep='\n')
        if dtype:
            return array.astype(dtype)
        else:
            return array

    def load_parameters(self):  # pragma: no cover
        """Depreciated in favor of ``mapdl.parameters``"""
        raise NotImplementedError('``load_parameters`` is  Depreciated.  '
                                  '\n\nInstead, please use:\n'
                                  '``mapdl.parameters``')

    def _display_plot(self, text):
        """Display the last generated plot (*.png) from MAPDL"""
        import scooby

        self._enable_interactive_plotting()
        png_found = PNG_TEST.findall(text)
        if png_found:
            # flush graphics writer
            self.show('CLOSE', mute=True)
            self.show('PNG', mute=True)

            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            filename = self._screenshot_path()

            if os.path.isfile(filename):
                img = mpimg.imread(filename)
                plt.imshow(img)
                plt.axis('off')
                if self._show_matplotlib_figures:  # pragma: no cover
                    plt.show()  # consider in-line plotting
                if scooby.in_ipython():
                    from IPython.display import display
                    display(plt.gcf())
            else:  # pragma: no cover
                self._log.error('Unable to find screenshot at %s', filename)

    def _screenshot_path(self):
        """Return last filename based on the current jobname"""
        filenames = glob.glob(os.path.join(self.directory, f'{self.jobname}*.png'))
        filenames.sort()
        return filenames[-1]

    def _set_log_level(self, level):
        """alias for set_log_level"""
        self.set_log_level(level)

    def list(self, filename, ext=''):
        """Displays the contents of an external, coded file.

        APDL Command: ``/LIST``

        Parameters
        ----------
        fname : str
            File name and directory path. An unspecified directory
            path defaults to the working directory.

        ext : str, optional
            Filename extension
        """
        if hasattr(self, '_local'):  # gRPC
            if not self._local:
                return self._download_as_raw(filename).decode()

        path = pathlib.Path(filename)
        if path.parent != '.':
            path = os.path.join(self.directory, filename)

        path = str(path) + ext
        with open(path) as fid:
            return fid.read()
