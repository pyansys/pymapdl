"""Post-processing module using MAPDL interface"""
import re
import weakref

import numpy as np

from ansys.mapdl.core.plotting import general_plotter
from ansys.mapdl.core.errors import MapdlRuntimeError
from ansys.mapdl.core.misc import supress_logging


COMPONENT_STRESS_TYPE = ['X', 'Y', 'Z', 'XY', 'YZ', 'XZ']
PRINCIPAL_TYPE = ['1', '2', '3']
STRESS_TYPES = ['X', 'Y', 'Z', 'XY', 'YZ', 'XZ', '1', '2', '3', 'INT', 'EQV']
COMP_TYPE = ['X', 'Y', 'Z', 'SUM']
DISP_TYPE = ['X', 'Y', 'Z', 'NORM', 'ALL']
ROT_TYPE = ['X', 'Y', 'Z', 'ALL']


def check_result_loaded(func):
    """Verify a result has been loaded within MAPDL"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            raise MapdlRuntimeError('Either this is an invalid result type for '
                                    'this solution, or '
                                    'no results set has been loaded within MAPDL.\n'
                                    'Load a result set with:\n\n'
                                    '\tmapdl.post1()\n'
                                    '\tmapdl.set(1, 1)') from None

    return wrapper


def check_comp(component, allowed):
    if not isinstance(component, str):
        raise TypeError('Component must be a string')
    component = component.upper().strip()
    if component not in allowed:
        raise ValueError('Component %s not a valid type.  ' % component +
                         'Allowed items:\n%s' % str(allowed))
    return component


class PostProcessing():
    """Post-processing using an active MAPDL session"""

    def __init__(self, mapdl):
        """Initialize postprocessing instance"""
        from ansys.mapdl.core.mapdl import _MapdlCore
        if not isinstance(mapdl, _MapdlCore):  # pragma: no cover
            raise TypeError('Must be initialized using Mapdl instance')
        self._mapdl_weakref = weakref.ref(mapdl)
        self._set_loaded = False

    @property
    def _mapdl(self):
        """Return the weakly referenced instance of MAPDL"""
        return self._mapdl_weakref()

    @property
    def _log(self):
        """alias for mapdl log"""
        return self._mapdl._log

    def _set_log_level(self, level):
        """alias for mapdl._set_log_level"""
        return self._mapdl._set_log_level(level)

    @supress_logging
    def __repr__(self):
        info = 'PyMAPDL PostProcessing Instance\n'
        info += '\tActive Result File:    %s\n' % self.filename
        info += '\tNumber of result sets: %d\n' % self.nsets
        info += '\tCurrent load step:     %d\n' % self.load_step
        info += '\tCurrent sub step:      %d\n' % self.sub_step

        if self._mapdl.parameters.routine == 'POST1':
            info += '\n\n' + self._mapdl.set('LIST')
        else:
            info += '\n\n Enable routine POST1 to see a table of available results'

        return info

    @property
    def time_values(self):
        """Return an array of the time values for all result sets.

        Examples
        --------
        Get all the time values after loading POST1.

        >>> mapdl.post1()
        >>> mapdl.post_processing.time_values
        [75.00054133588232,
         75.00081189985094,
         75.00121680412036,
         75.00574491751847,
         75.03939292229019,
         75.20949687626468]
        """
        list_rsp = self._mapdl.set('LIST')
        groups = re.findall(r'([-+]?\d*\.\d+|\d+)', list_rsp)

        # values will always be the second set
        return np.array([float(item) for item in (groups[1::5])])

    def _reset_cache(self):
        """Reset local cache"""
        self._set_loaded = False

    @property
    def filename(self) -> str:
        """Return the current result file name without extension.

        Examples
        --------
        >>> mapdl.post_processing.filename
        'file'
        """
        response = self._mapdl.run('/INQUIRE, param, RSTFILE', mute=False)
        return response.split('=')[-1].strip()

    @property
    def nsets(self) -> int:
        """Number of data sets on result file.

        Examples
        >>> mapdl.post_processing.nsets
        1
        """
        return int(self._mapdl.get_value("ACTIVE", item1="SET", it1num='NSET'))

    @property
    def load_step(self) -> int:
        """Current load step number

        Examples
        --------
        >>> mapdl.post1()
        >>> mapdl.set(2, 2)
        >>> mapdl.post_processing.load_step
        2
        """
        return int(self._mapdl.get_value("ACTIVE", item1="SET", it1num='LSTP'))

    @property
    def sub_step(self) -> int:
        """Current sub step number

        Examples
        --------
        >>> mapdl.post1()
        >>> mapdl.set(2, 2)
        >>> mapdl.post_processing.load_step
        2
        """
        return int(self._mapdl.get_value("ACTIVE", item1="SET", it1num='SBST'))

    @property
    def time(self) -> float:
        """Time associated with current result in the database.

        Examples
        --------
        Time of the current result of a modal analysis

        >>> mapdl.post1()
        >>> mapdl.set(1, 1)
        >>> mapdl.post_processing.time
        1.0
        """
        return self._mapdl.get_value("ACTIVE", item1="SET", it1num='TIME')

    @property
    def freq(self) -> float:
        """Freqneyc associated with current result in the database.

        Applicable for a Modal, harmonic or spectral analysis.

        Examples
        --------
        Natural frequency of the current result of a modal analysis

        >>> mapdl.post1()
        >>> mapdl.set(1, 1)
        >>> mapdl.post_processing.freq
        956.86239847
        """
        return self._mapdl.get_value("ACTIVE", item1="SET", it1num='FREQ')

    def nodal_displacement(self, component='NORM') -> np.ndarray:
        """Nodal X, Y, or Z structural displacement

        Equilvanent MAPDL command:
        ``PRNSOL, U, X``

        Parameters
        ----------
        component : str, optional
            Structural displacement component to retrieve.  Must be
            ``'X'``, ``'Y'``, ``'Z'``, ``'ALL'``, or ``'NORM'``.
            Defaults to ``'NORM'``.

        Examples
        --------
        >>> mapdl.post_processing.nodal_displacement('X')
        array([1.07512979e-04, 8.59137773e-05, 5.70690047e-05, ...,
               5.70333124e-05, 8.58600402e-05, 1.07445726e-04])

        Displacement in all dimensions

        >>> mapdl.post_processing.nodal_displacement('ALL')
        array([[ 1.07512979e-04,  6.05382076e-05, -1.64333622e-11],
               [ 8.59137773e-05,  7.88053970e-05, -1.93668243e-11],
               [ 5.70690047e-05,  1.23100157e-04, -1.04703715e-11],
               ...,
               [ 5.70333124e-05,  1.23023176e-04, -9.77598660e-12],
               [ 8.58600402e-05,  7.87561008e-05, -9.12531408e-12],
               [ 1.07445726e-04,  6.05003408e-05, -1.23634647e-11]])

        Nodes corresponding to the nodal displacements

        >>> mapdl.mesh.nnum_all
        array([   1,    2,    3, ..., 7215, 7216, 7217], dtype=int32)

        Notes
        -----
        This command always returns all nodal displacements regardless
        of if the nodes are selected or not.

        """
        component = check_comp(component, DISP_TYPE)

        if component in ['NORM', 'ALL']:
            x = self._ndof_rst('U', 'X')
            y = self._ndof_rst('U', 'Y')
            z = self._ndof_rst('U', 'Z')
            disp = np.vstack((x, y, z))
            if component == 'NORM':
                return np.linalg.norm(disp, axis=0)
            return disp.T

        return self._ndof_rst('U', component)

    def plot_nodal_displacement(self, component='NORM', show_node_numbering=False,
                                **kwargs):
        """Plot nodal displacement

        Parameters
        ----------
        component : str, optional
            Structural displacement component to retrieve.  Must be
            ``'X'``, ``'Y'``, ``'Z'``, or ``'NORM'``.  Defaults to
            ``'NORM'``.

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the normalized nodal displacement for the second result

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_displacement('NORM',
                                                          smooth_shading=True)

        Plot the x displacement without smooth shading with individual
        node numbering

        >>> mapdl.post_processing.plot_nodal_displacement('X',
                                                          show_node_numbering=True)
        """
        if isinstance(component, str):
            if component.upper() == 'ALL':
                raise ValueError('"ALL" not allowed in this context.  Select a '
                                 'single displacement component (e.g. "X")')

        disp = self.nodal_displacement(component)
        kwargs.setdefault('stitle', '%s Displacement' % component)
        return self._plot_point_scalars(disp, show_node_numbering=show_node_numbering,
                                        **kwargs)

    def _plot_point_scalars(self, scalars, show_node_numbering=False, **kwargs):
        """Plot point scalars

        Assumes scalars are from all nodes and not just the active surface.
        """
        surf = self._mapdl.mesh._surf

        # as ``disp`` returns the result for all nodes, we need all node numbers
        # and to index to the output node numbers
        if hasattr(self._mapdl.mesh, 'nnum_all'):
            nnum = self._mapdl.mesh.nnum_all
        else:
            nnum = self._all_nnum

        mask = np.in1d(nnum, surf['ansys_node_num'])
        ridx = np.argsort(np.argsort(surf['ansys_node_num']))
        if scalars.size != mask.size:
            scalars = scalars[self.selected_nodes]
        scalars = scalars[mask][ridx]

        meshes = [{'mesh': surf.copy(deep=False),  # deep=False for ipyvtk-simple
                   'stitle': kwargs.pop('stitle', ''),
                   'scalars': scalars}]

        labels = []
        if show_node_numbering:
            labels = [{'points': surf.points, 'labels': surf['ansys_node_num']}]

        kwargs.setdefault('title', 'MAPDL Displacement')
        return general_plotter(meshes, [], labels, **kwargs)

    @property
    @supress_logging
    def _all_nnum(self):
        self._mapdl.cm('__TMP_NODE__', 'NODE')
        self._mapdl.allsel()
        nnum = self._mapdl.get_array('NODE', item1='NLIST').astype(np.int32)
        if nnum[0] == -1:
            nnum = self._mapdl.get_array('NODE', item1='NLIST').astype(np.int32)
        self._mapdl.cmsel('S', '__TMP_NODE__', 'NODE')
        return nnum

    @property
    def _nsel(self):
        """Return the ANSYS formatted selected nodes array.

        -1 for unselected
        0 for undefined
        1 for selected

        """
        return self._ndof_rst('NSEL').astype(np.int8)

    @property
    def selected_nodes(self) -> np.ndarray:
        """Mask of the selected nodes.

        Examples
        --------
        >>> mapdl.post_processing.node_selection
        array([False, False, False, ..., True, True, True])

        """
        return self._nsel == 1

    def nodal_rotation(self, component='ALL') -> np.ndarray:
        """Nodal X, Y, or Z structural rotation

        Equilvanent MAPDL commands:
        ``PRNSOL, ROT, X``
        ``PRNSOL, ROT, Y``
        ``PRNSOL, ROT, Z``

        Parameters
        ----------
        component : str, optional
            Structural rotational component to retrieve.  Must be
            ``'X'``, ``'Y'``, ``'Z'``, ``'ALL'``.  Defaults to ``'ALL'``

        Examples
        --------
        Nodal rotation in all dimensions for current result

        >>> mapdl.post1()
        >>> mapdl.set(1, 1)
        >>> mapdl.post_processing.nodal_rotation('ALL')
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               ...,
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])

        Nodes corresponding to the nodal rotations

        >>> mapdl.mesh.nnum_all
        array([   1,    2,    3, ..., 7215, 7216, 7217], dtype=int32)

        Notes
        -----
        This command always returns all nodal rotations regardless of
        if the nodes are selected or not.  Use the ``selected_nodes``
        mask to get the currently selected nodes.
        """
        component = check_comp(component, ROT_TYPE)

        if component == 'ALL':
            x = self._ndof_rst('ROT', 'X')
            y = self._ndof_rst('ROT', 'Y')
            z = self._ndof_rst('ROT', 'Z')
            return np.vstack((x, y, z)).T

        return self._ndof_rst('ROT', component)

    def plot_nodal_rotation(self, component, show_node_numbering=False,
                            **kwargs):
        """Plot nodal rotation.

        Parameters
        ----------
        component : str
            Structural rotation component to retrieve.  Must be
            ``'X'``, ``'Y'``, or ``'Z'``.

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the x rotation without smooth shading with individual
        node numbering

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_rotation('X', show_node_numbering=True)
        """
        if isinstance(component, str):
            if component.upper() == 'ALL':
                raise ValueError('"ALL" not allowed in this context.  Select a '
                                 'single component (e.g. "X")')

        disp = self.nodal_rotation(component)
        kwargs.setdefault('stitle', '%s Rotation' % component)
        return self._plot_point_scalars(disp, show_node_numbering=show_node_numbering,
                                        **kwargs)

    @check_result_loaded
    def _ndof_rst(self, item, it1num=''):
        """Nodal degree of freedom result"""
        return self._mapdl.get_array('NODE', item1=item, it1num=it1num)

    @property
    def nodal_temperature(self) -> np.ndarray:
        """The nodal temperature of the current result.

        Equilvanent MAPDL command:
        ``PRNSOL, TEMP``

        Examples
        --------
        >>> mapdl.post_processing.temperature
        array([0., 0., 0., ..., 0., 0., 0.])

        Notes
        -----
        The nodal results are averaged across all selected elements.
        Not all nodes will contain valid results (e.g. midside nodes),
        and those nodes will report a zero value.

        Elements that are not selected will not contribute to the
        averaged nodal values, and if a node's attached elements are
        all unselected, the element will report a zero value.
        """
        return self._ndof_rst('TEMP')

    def plot_nodal_temperature(self, show_node_numbering=False, **kwargs):
        """Plot nodal temperature of the current result.

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the nodal temperature for the second result

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.temperature()

        Plot off_screen and save a screenshot

        >>> mapdl.post_processing.plot_nodal_temperature(off_screen=True,
                                                         screenshot='temp_1_2.png')

        Subselect a single result type and plot those stress results
        >>> mapdl.esel('S', 'TYPE', vmin=1)
        >>> mapdl.post_processing.plot_nodal_temperature(smooth_shading=True)
        """
        kwargs.setdefault('stitle', 'Nodal\nTemperature')
        return self._plot_point_scalars(self.nodal_temperature,
                                        show_node_numbering=show_node_numbering,
                                        **kwargs)

    @property
    def nodal_pressure(self) -> np.ndarray:
        """The nodal pressure of the current result.

        Equilvanent MAPDL command:
        ``PRNSOL, PRES``

        Examples
        --------
        >>> mapdl.post_processing.pressure
        array([0., 0., 0., ..., 0., 0., 0.])

        Notes
        -----
        The nodal results are averaged across all selected elements.
        Not all nodes will contain valid results (e.g. midside nodes),
        and those nodes will report a zero value.

        Elements that are not selected will not contribute to the
        averaged nodal values, and if a node's attached elements are
        all unselected, the element will report a zero value.
        """
        return self._ndof_rst('PRES')

    def plot_nodal_pressure(self, show_node_numbering=False, **kwargs):
        """Plot nodal pressure of the current result.

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the nodal pressure for the second result

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_pressure()

        Plot off_screen and save a screenshot

        >>> mapdl.post_processing.plot_nodal_pressure(off_screen=True,
                                                   screenshot='temp_1_2.png')

        Subselect a single result type and plot those stress results
        >>> mapdl.esel('S', 'TYPE', vmin=1)
        >>> mapdl.post_processing.plot_nodal_pressure(smooth_shading=True)
        """
        kwargs.setdefault('stitle', 'Nodal\nPressure')
        return self._plot_point_scalars(self.nodal_pressure,
                                        show_node_numbering=show_node_numbering,
                                        **kwargs)

    @property
    def nodal_voltage(self) -> np.ndarray:
        """The nodal voltage of the current result.

        Equilvanent MAPDL command:
        ``PRNSOL, PRES``

        Examples
        --------
        >>> mapdl.post_processing.voltage
        array([0., 0., 0., ..., 0., 0., 0.])

        Notes
        -----
        The nodal results are averaged across all selected elements.
        Not all nodes will contain valid results (e.g. midside nodes),
        and those nodes will report a zero value.

        Elements that are not selected will not contribute to the
        averaged nodal values, and if a node's attached elements are
        all unselected, the element will report a zero value.
        """
        return self._ndof_rst('VOLT')

    def plot_nodal_voltage(self, show_node_numbering=False, **kwargs):
        """Plot nodal voltage of the current result.

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the nodal voltage for the second result

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_voltage()

        Plot off_screen and save a screenshot

        >>> mapdl.post_processing.plot_nodal_voltage(off_screen=True,
                                                   screenshot='temp_1_2.png')

        Subselect a single result type and plot those stress results
        >>> mapdl.esel('S', 'TYPE', vmin=1)
        >>> mapdl.post_processing.plot_nodal_voltage(smooth_shading=True)
        """
        kwargs.setdefault('stitle', 'Nodal\nVoltage')
        return self._plot_point_scalars(self.nodal_voltage,
                                        show_node_numbering=show_node_numbering,
                                        **kwargs)

    def nodal_component_stress(self, component) -> np.ndarray:
        """Nodal component stress.

        Equilvanent MAPDL commands:
        \*VGET, PARM, NODE, , S, X
        PRNSOL, S, COMP

        Parameters
        ----------
        component : str, optional
            Nodal component stress component to retrieve.  Must be
            ``'X'``, ``'Y'``, ``'Z'``, ``'XY'``, ``'YZ'``, or
            ``'XZ'``.

        Examples
        --------
        Nodal stress in the X direction for the first result

        >>> mapdl.post1()
        >>> mapdl.set(1, 1)
        >>> mapdl.post_processing.nodal_component_stress('X')
            array([0.60024621, 0.61625265, 0.65081825, ...,
                   0.        , 0.        , 0.        ])

        Corresponding nodes

        >>> mapdl.mesh.nnum_all
        array([   1,    2,    3, ..., 7215, 7216, 7217], dtype=int32)

        Notes
        -----
        This command always returns all nodal rotations regardless of
        if the nodes are selected or not.  Use the ``selected_nodes``
        mask to get the currently selected nodes.
        """
        component = check_comp(component, COMPONENT_STRESS_TYPE)
        return self._ndof_rst('S', component)

    def plot_nodal_component_stress(self, component, show_node_numbering=False,
                                    **kwargs):
        """Plot nodal component stress.

        Parameters
        ----------
        component : str
            Nodal component stress component to plot.  Must be
            ``'X'``, ``'Y'``, ``'Z'``, ``'XY'``, ``'YZ'``, or
            ``'XZ'``.

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the x nodal component stress for the second result set

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_component_stress('X')
        """
        disp = self.nodal_component_stress(component)
        kwargs.setdefault('stitle', '%s Nodal\nStress' % component)
        return self._plot_point_scalars(disp, show_node_numbering=show_node_numbering,
                                        **kwargs)

    def nodal_principal_stress(self, component) -> np.ndarray:
        """Nodal principal stress.

        Equilvanent MAPDL commands:
        \*VGET, PARM, NODE, , S, 1
        PRNSOL, S, PRIN

        Parameters
        ----------
        component : str, optional
            Nodal component stress component to retrieve.  Must be
            ``'1'``, ``'2'``, or ``'3'``

        Examples
        --------
        Nodal stress in the S1 direction for the first result

        >>> mapdl.post1()
        >>> mapdl.set(1, 1)
        >>> mapdl.post_processing.nodal_principal_stress('1')
            array([0.60024621, 0.61625265, 0.65081825, ...,
                   0.        , 0.        , 0.        ])

        Corresponding nodes

        >>> mapdl.mesh.nnum_all
        array([   1,    2,    3, ..., 7215, 7216, 7217], dtype=int32)

        Notes
        -----
        This command always returns all nodal rotations regardless of
        if the nodes are selected or not.  Use the ``selected_nodes``
        mask to get the currently selected nodes.
        """
        if isinstance(component, int):
            component = str(component)
        component = check_comp(component, PRINCIPAL_TYPE)
        return self._ndof_rst('S', component)

    def plot_nodal_principal_stress(self, component, show_node_numbering=False,
                                    **kwargs):
        """Plot nodal principal stress.

        Parameters
        ----------
        component : str
            Nodal component stress component to plot.  Must be
            ``'1'``, ``'2'``, or ``'3'``

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the nodal principal stress "1" for the second result set

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_component_stress('1')
        """
        disp = self.nodal_principal_stress(component)
        kwargs.setdefault('stitle', '%s Nodal\nPrincipal Stress' % component)
        return self._plot_point_scalars(disp, show_node_numbering=show_node_numbering,
                                        **kwargs)

    @property
    def nodal_stress_intensity(self) -> np.ndarray:
        """The nodal stress intensity of the current result.

        Equilvanent MAPDL command:
        ``PRNSOL, S, PRIN``

        Examples
        --------
        Stress intensity for result 2

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.nodal_stress_intensity
        array([15488.84357602, 16434.95432337, 15683.2334295 , ...,
                   0.        ,     0.        ,     0.        ])

        Notes
        -----
        The nodal results are averaged across all selected elements.
        Not all nodes will contain valid results (e.g. midside nodes),
        and those nodes will report a zero stress.

        Elements that are not selected will not contribute to the
        averaged nodal values, and if a node's attached elements are
        all unselected, the element will report a zero stress value.
        """
        return self._ndof_rst('S', 'INT')

    def plot_nodal_stress_intensity(self, show_node_numbering=False, **kwargs):
        """Plot the nodal stress intensity of the current result.

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the equivalent stress for the second result

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_stress_intensity()

        Plot off_screen and save a screenshot

        >>> mapdl.post_processing.plot_nodal_stress_intensity(off_screen=True,
                                                              screenshot='seqv_00.png')

        Subselect a single result type and plot those stress results
        >>> mapdl.esel('S', 'TYPE', vmin=1)
        >>> mapdl.post_processing.plot_nodal_stress_intensity(smooth_shading=True)

        """
        scalars = self.nodal_stress_intensity
        kwargs.setdefault('stitle', 'Nodal Stress\nIntensity')
        return self._plot_point_scalars(scalars,
                                        show_node_numbering=show_node_numbering,
                                        **kwargs)

    @property
    def nodal_eqv_stress(self) -> np.ndarray:
        """The nodal equivalent stress of the current result.

        Equilvanent MAPDL command:
        ``PRNSOL, S, PRIN``

        Examples
        --------
        >>> mapdl.post_processing.nodal_eqv_stress
        array([15488.84357602, 16434.95432337, 15683.2334295 , ...,
                   0.        ,     0.        ,     0.        ])

        Stress from result 2

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.nodal_eqv_stress
        array([15488.84357602, 16434.95432337, 15683.2334295 , ...,
                   0.        ,     0.        ,     0.        ])

        Notes
        -----
        The nodal results are averaged across all selected elements.
        Not all nodes will contain valid results (e.g. midside nodes),
        and those nodes will report a zero stress.

        Elements that are not selected will not contribute to the
        averaged nodal values, and if a node's attached elements are
        all unselected, the element will report a zero stress value.

        """
        return self._ndof_rst('S', 'EQV')

    def plot_nodal_eqv_stress(self, show_node_numbering=False, **kwargs):
        """Plot nodal equivalent stress of the current result.

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the equivalent stress for the second result

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_eqv_stress()

        Plot off_screen and save a screenshot

        >>> mapdl.post_processing.plot_nodal_eqv_stress(off_screen=True,
                                                        screenshot='seqv_00.png')

        Subselect a single result type and plot those stress results
        >>> mapdl.esel('S', 'TYPE', vmin=1)
        >>> mapdl.post_processing.plot_nodal_eqv_stress(smooth_shading=True)

        """
        scalars = self.nodal_eqv_stress
        kwargs.setdefault('stitle', 'Nodal Equilvanent\nStress')
        return self._plot_point_scalars(scalars,
                                        show_node_numbering=show_node_numbering,
                                        **kwargs)

    def nodal_total_component_strain(self, component) -> np.ndarray:
        """Total nodal component strain

        Includes elastic, plastic, and creep strain.

        Equilvanent MAPDL commands:
        \*VGET, PARM, NODE, , EPTO, X

        Parameters
        ----------
        component : str, optional
            Component to retrieve.  Must be ``'X'``, ``'Y'``, ``'Z'``,
            ``'XY'``, ``'YZ'``, or ``'XZ'``.

        Examples
        --------
        Total component strain in the X direction for the first result

        >>> mapdl.post1()
        >>> mapdl.set(1, 1)
        >>> mapdl.post_processing.nodal_total_component_strain('X')
            array([0.60024621, 0.61625265, 0.65081825, ...,
                   0.        , 0.        , 0.        ])

        Corresponding nodes

        >>> mapdl.mesh.nnum_all
        array([   1,    2,    3, ..., 7215, 7216, 7217], dtype=int32)

        Notes
        -----
        This command always returns all nodal rotations regardless of
        if the nodes are selected or not.  Use the ``selected_nodes``
        mask to get the currently selected nodes.
        """
        if isinstance(component, int):
            component = str(component)
        component = check_comp(component, COMPONENT_STRESS_TYPE)
        return self._ndof_rst('EPTO', component)

    def plot_nodal_total_component_strain(self, component, show_node_numbering=False,
                                          **kwargs):
        """Plot nodal principal stress.

        Includes elastic, plastic, and creep strain.

        Parameters
        ----------
        component : str
            Nodal component stress component to plot.  Must be
            ``'1'``, ``'2'``, or ``'3'``

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the nodal principal stress "1" for the second result set

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_component_stress('1')
        """
        disp = self.nodal_total_component_strain(component)
        kwargs.setdefault('stitle', '%s Total Nodal\nComponent Strain' % component)
        return self._plot_point_scalars(disp, show_node_numbering=show_node_numbering,
                                        **kwargs)

    def nodal_total_principal_strain(self, component) -> np.ndarray:
        """Total nodal principal total strain.

        Includes elastic, plastic, and creep strain.

        Equilvanent MAPDL commands:
        \*VGET, PARM, NODE, , EPTO, 1

        Parameters
        ----------
        component : str, optional
            Component to retrieve.  Must be ``'1'``, ``'2'``, or
            ``'3'``

        Examples
        --------
        Principal nodal strain in the S1 direction for the first result

        >>> mapdl.post1()
        >>> mapdl.set(1, 1)
        >>> mapdl.post_processing.nodal_total_principal_strain('1')
            array([0.60024621, 0.61625265, 0.65081825, ...,
                   0.        , 0.        , 0.        ])

        Corresponding nodes

        >>> mapdl.mesh.nnum_all
        array([   1,    2,    3, ..., 7215, 7216, 7217], dtype=int32)

        Notes
        -----
        This command always returns all nodal rotations regardless of
        if the nodes are selected or not.  Use the ``selected_nodes``
        mask to get the currently selected nodes.
        """
        if isinstance(component, int):
            component = str(component)
        component = check_comp(component, PRINCIPAL_TYPE)
        return self._ndof_rst('EPTO', component)

    def plot_nodal_total_principal_strain(self, component,
                                          show_node_numbering=False,
                                          **kwargs):
        """Plot total nodal principal strain.

        Includes elastic, plastic, and creep strain.

        Parameters
        ----------
        component : str
            Nodal principal strain component to plot.  Must be
            ``'1'``, ``'2'``, or ``'3'``

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the nodal principal stress "1" for the second result set

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_component_stress('1')
        """
        disp = self.nodal_total_principal_strain(component)
        kwargs.setdefault('stitle', '%s Nodal\nPrincipal Strain' % component)
        return self._plot_point_scalars(disp, show_node_numbering=show_node_numbering,
                                        **kwargs)

    @property
    def nodal_total_strain_intensity(self) -> np.ndarray:
        """The total nodal strain intensity of the current result.

        Equilvanent MAPDL command:
        ``PRNSOL, EPTO, PRIN``

        Examples
        --------
        Total strain intensity for result 2

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.nodal_total_strain_intensity
        array([15488.84357602, 16434.95432337, 15683.2334295 , ...,
                   0.        ,     0.        ,     0.        ])

        Notes
        -----
        The nodal results are averaged across all selected elements.
        Not all nodes will contain valid results (e.g. midside nodes),
        and those nodes will report a zero stress.

        Elements that are not selected will not contribute to the
        averaged nodal values, and if a node's attached elements are
        all unselected, the element will report a zero stress value.
        """
        return self._ndof_rst('EPEL', 'INT')

    def plot_nodal_total_strain_intensity(self,
                                          show_node_numbering=False,
                                          **kwargs):
        """Plot the total nodal strain intensity of the current result.

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the total strain intensity for the second result

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_total_strain_intensity()

        Plot off_screen and save a screenshot

        >>> mapdl.post_processing.plot_nodal_total_strain_intensity(off_screen=True,
                                                                    screenshot='seqv_00.png')

        Subselect a single result type and plot those strain results
        >>> mapdl.esel('S', 'TYPE', vmin=1)
        >>> mapdl.post_processing.plot_nodal_total_strain_intensity()

        """
        scalars = self.nodal_total_strain_intensity
        kwargs.setdefault('stitle', 'Total Nodal\nStrain Intensity')
        return self._plot_point_scalars(scalars,
                                        show_node_numbering=show_node_numbering,
                                        **kwargs)

    @property
    def nodal_total_eqv_strain(self) -> np.ndarray:
        """The total nodal equivalent strain of the current result.

        Equilvanent MAPDL command:
        ``PRNSOL, EPTO, PRIN``

        Examples
        --------
        Total quivalent strain for the current result

        >>> mapdl.post_processing.nodal_total_eqv_strain
        array([15488.84357602, 16434.95432337, 15683.2334295 , ...,
                   0.        ,     0.        ,     0.        ])

        Strain from result 2

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.nodal_total_eqv_strain
        array([15488.84357602, 16434.95432337, 15683.2334295 , ...,
                   0.        ,     0.        ,     0.        ])

        Notes
        -----
        The nodal results are averaged across all selected elements.
        Not all nodes will contain valid results (e.g. midside nodes),
        and those nodes will report a zero stress.

        Elements that are not selected will not contribute to the
        averaged nodal values, and if a node's attached elements are
        all unselected, the element will report a zero stress value.
        """
        return self._ndof_rst('EPTO', 'EQV')

    def plot_nodal_total_eqv_strain(self, show_node_numbering=False, **kwargs):
        """Plot the total nodal equivalent strain of the current result.

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the total equivalent strain for the second result

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_total_eqv_strain()

        Plot off_screen and save a screenshot

        >>> mapdl.post_processing.plot_nodal_total_eqv_strain(off_screen=True,
                                                              screenshot='seqv_00.png')

        Subselect a single result type and plot those strain results
        >>> mapdl.esel('S', 'TYPE', vmin=1)
        >>> mapdl.post_processing.plot_nodal_total_eqv_strain(smooth_shading=True)

        """
        scalars = self.nodal_total_eqv_strain
        kwargs.setdefault('stitle', 'Total Nodal\nEquivalent Strain')
        return self._plot_point_scalars(scalars,
                                        show_node_numbering=show_node_numbering,
                                        **kwargs)
###############################################################################


    def nodal_elastic_component_strain(self, component) -> np.ndarray:
        """Elastic nodal component strain

        Equivalent MAPDL command:
        PRNSOL, EPEL, PRIN

        Parameters
        ----------
        component : str, optional
            Component to retrieve.  Must be ``'X'``, ``'Y'``, ``'Z'``,
            ``'XY'``, ``'YZ'``, or ``'XZ'``.

        Examples
        --------
        Elastic component strain in the X direction for the first result

        >>> mapdl.post1()
        >>> mapdl.set(1, 1)
        >>> mapdl.post_processing.nodal_elastic_component_strain('X')
            array([0.60024621, 0.61625265, 0.65081825, ...,
                   0.        , 0.        , 0.        ])

        Corresponding nodes

        >>> mapdl.mesh.nnum_all
        array([   1,    2,    3, ..., 7215, 7216, 7217], dtype=int32)

        Notes
        -----
        This command always returns all nodal rotations regardless of
        if the nodes are selected or not.  Use the ``selected_nodes``
        mask to get the currently selected nodes.
        """
        if isinstance(component, int):
            component = str(component)
        component = check_comp(component, COMPONENT_STRESS_TYPE)
        return self._ndof_rst('EPEL', component)

    def plot_nodal_elastic_component_strain(self, component, show_node_numbering=False,
                                            **kwargs):
        """Plot nodal elastic component strain.

        Parameters
        ----------
        component : str
            Nodal elastic component to plot.  Must be ``'X'``,
            ``'Y'``, ``'Z'``, ``'XY'``, ``'YZ'``, or ``'XZ'``.

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the nodal elastic principal strain "1" for the second result set

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_elastic_component_strain('1')
        """
        disp = self.nodal_elastic_component_strain(component)
        kwargs.setdefault('stitle', '%s Elastic Nodal\nComponent Strain' % component)
        return self._plot_point_scalars(disp, show_node_numbering=show_node_numbering,
                                        **kwargs)

    def nodal_elastic_principal_strain(self, component) -> np.ndarray:
        """Nodal elastic principal elastic strain.

        Equivalent MAPDL commands:
        \*VGET, PARM, NODE, , EPEL, 1

        Parameters
        ----------
        component : str, optional
            Component to retrieve.  Must be ``'1'``, ``'2'``, or
            ``'3'``

        Examples
        --------
        Principal nodal strain in the S1 direction for the first result

        >>> mapdl.post1()
        >>> mapdl.set(1, 1)
        >>> mapdl.post_processing.nodal_elastic_principal_strain('1')
            array([0.60024621, 0.61625265, 0.65081825, ...,
                   0.        , 0.        , 0.        ])

        Corresponding nodes

        >>> mapdl.mesh.nnum_all
        array([   1,    2,    3, ..., 7215, 7216, 7217], dtype=int32)

        Notes
        -----
        This command always returns all nodal rotations regardless of
        if the nodes are selected or not.  Use the ``selected_nodes``
        mask to get the currently selected nodes.
        """
        if isinstance(component, int):
            component = str(component)
        component = check_comp(component, PRINCIPAL_TYPE)
        return self._ndof_rst('EPEL', component)

    def plot_nodal_elastic_principal_strain(self, component,
                                            show_node_numbering=False,
                                            **kwargs):
        """Plot elastic nodal principal strain.

        Parameters
        ----------
        component : str
            Nodal principal strain component to plot.  Must be
            ``'1'``, ``'2'``, or ``'3'``

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the nodal principal strain "1" for the second result set

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_elastic_principal_strain('1')
        """
        disp = self.nodal_elastic_principal_strain(component)
        kwargs.setdefault('stitle', '%s Nodal\nPrincipal Strain' % component)
        return self._plot_point_scalars(disp, show_node_numbering=show_node_numbering,
                                        **kwargs)

    @property
    def nodal_elastic_strain_intensity(self) -> np.ndarray:
        """The elastic nodal strain intensity of the current result.

        Equivalent MAPDL command:
        ``PRNSOL, EPEL, PRIN``

        Examples
        --------
        Elastic strain intensity for result 2

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.nodal_elastic_strain_intensity
        array([15488.84357602, 16434.95432337, 15683.2334295 , ...,
                   0.        ,     0.        ,     0.        ])

        Notes
        -----
        The nodal results are averaged across all selected elements.
        Not all nodes will contain valid results (e.g. midside nodes),
        and those nodes will report a zero value.

        Elements that are not selected will not contribute to the
        averaged nodal values, and if a node's attached elements are
        all unselected, the element will report a zero value.
        """
        return self._ndof_rst('EPEL', 'INT')

    def plot_nodal_elastic_strain_intensity(self,
                                            show_node_numbering=False,
                                            **kwargs):
        """Plot the elastic nodal strain intensity of the current result.

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the elastic strain intensity for the second result

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_elastic_strain_intensity()

        Plot off_screen and save a screenshot

        >>> mapdl.post_processing.plot_nodal_elastic_strain_intensity(off_screen=True,
                                                                    screenshot='seqv_00.png')

        Subselect a single result type and plot those strain results
        >>> mapdl.esel('S', 'TYPE', vmin=1)
        >>> mapdl.post_processing.plot_nodal_elastic_strain_intensity()

        """
        scalars = self.nodal_elastic_strain_intensity
        kwargs.setdefault('stitle', 'Elastic Nodal\nStrain Intensity')
        return self._plot_point_scalars(scalars,
                                        show_node_numbering=show_node_numbering,
                                        **kwargs)

    @property
    def nodal_elastic_eqv_strain(self) -> np.ndarray:
        """The elastic nodal equivalent strain of the current result.

        Equivalent MAPDL command:
        ``PRNSOL, EPEL, PRIN``

        Examples
        --------
        Elastic quivalent strain for the current result

        >>> mapdl.post_processing.nodal_elastic_eqv_strain
        array([15488.84357602, 16434.95432337, 15683.2334295 , ...,
                   0.        ,     0.        ,     0.        ])

        Strain from result 2

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.nodal_elastic_eqv_strain
        array([15488.84357602, 16434.95432337, 15683.2334295 , ...,
                   0.        ,     0.        ,     0.        ])

        Notes
        -----
        The nodal results are averaged across all selected elements.
        Not all nodes will contain valid results (e.g. midside nodes),
        and those nodes will report a zero value.

        Elements that are not selected will not contribute to the
        averaged nodal values, and if a node's attached elements are
        all unselected, the element will report a zero value.
        """
        return self._ndof_rst('EPEL', 'EQV')

    def plot_nodal_elastic_eqv_strain(self, show_node_numbering=False, **kwargs):
        """Plot the elastic nodal equivalent strain of the current result.

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the elastic equivalent strain for the second result

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_elastic_eqv_strain()

        Plot off_screen and save a screenshot

        >>> mapdl.post_processing.plot_nodal_elastic_eqv_strain(off_screen=True,
                                                              screenshot='seqv_00.png')

        Subselect a single result type and plot those strain results
        >>> mapdl.esel('S', 'TYPE', vmin=1)
        >>> mapdl.post_processing.plot_nodal_elastic_eqv_strain(smooth_shading=True)

        """
        scalars = self.nodal_elastic_eqv_strain
        kwargs.setdefault('stitle', 'Elastic Nodal\n Equivalent Strain')
        return self._plot_point_scalars(scalars,
                                        show_node_numbering=show_node_numbering,
                                        **kwargs)


###############################################################################


    def nodal_plastic_component_strain(self, component) -> np.ndarray:
        """Plastic nodal component strain

        Equivalent MAPDL command:
        PRNSOL, EPPL, PRIN

        Parameters
        ----------
        component : str, optional
            Component to retrieve.  Must be ``'X'``, ``'Y'``, ``'Z'``,
            ``'XY'``, ``'YZ'``, or ``'XZ'``.

        Examples
        --------
        Plastic component strain in the X direction for the first result

        >>> mapdl.post1()
        >>> mapdl.set(1, 1)
        >>> mapdl.post_processing.nodal_plastic_component_strain('X')
            array([0.60024621, 0.61625265, 0.65081825, ...,
                   0.        , 0.        , 0.        ])

        Corresponding nodes

        >>> mapdl.mesh.nnum_all
        array([   1,    2,    3, ..., 7215, 7216, 7217], dtype=int32)

        Notes
        -----
        This command always returns all nodal rotations regardless of
        if the nodes are selected or not.  Use the ``selected_nodes``
        mask to get the currently selected nodes.
        """
        if isinstance(component, int):
            component = str(component)
        component = check_comp(component, COMPONENT_STRESS_TYPE)
        return self._ndof_rst('EPPL', component)

    def plot_nodal_plastic_component_strain(self, component, show_node_numbering=False,
                                            **kwargs):
        """Plot nodal plastic component strain.

        Parameters
        ----------
        component : str
            Nodal plastic component to plot.  Must be ``'X'``,
            ``'Y'``, ``'Z'``, ``'XY'``, ``'YZ'``, or ``'XZ'``.

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the nodal plastic principal strain "1" for the second result set

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_plastic_component_strain('1')
        """
        disp = self.nodal_plastic_component_strain(component)
        kwargs.setdefault('stitle', '%s Plastic Nodal\nComponent Strain' % component)
        return self._plot_point_scalars(disp, show_node_numbering=show_node_numbering,
                                        **kwargs)

    def nodal_plastic_principal_strain(self, component) -> np.ndarray:
        """Nodal plastic principal plastic strain.

        Equivalent MAPDL commands:
        \*VGET, PARM, NODE, , EPPL, 1

        Parameters
        ----------
        component : str, optional
            Component to retrieve.  Must be ``'1'``, ``'2'``, or
            ``'3'``

        Examples
        --------
        Principal nodal strain in the S1 direction for the first result

        >>> mapdl.post1()
        >>> mapdl.set(1, 1)
        >>> mapdl.post_processing.nodal_plastic_principal_strain('1')
            array([0.60024621, 0.61625265, 0.65081825, ...,
                   0.        , 0.        , 0.        ])

        Corresponding nodes

        >>> mapdl.mesh.nnum_all
        array([   1,    2,    3, ..., 7215, 7216, 7217], dtype=int32)

        Notes
        -----
        This command always returns all nodal rotations regardless of
        if the nodes are selected or not.  Use the ``selected_nodes``
        mask to get the currently selected nodes.
        """
        if isinstance(component, int):
            component = str(component)
        component = check_comp(component, PRINCIPAL_TYPE)
        return self._ndof_rst('EPPL', component)

    def plot_nodal_plastic_principal_strain(self, component,
                                            show_node_numbering=False,
                                            **kwargs):
        """Plot plastic nodal principal strain.

        Parameters
        ----------
        component : str
            Nodal principal strain component to plot.  Must be
            ``'1'``, ``'2'``, or ``'3'``

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the nodal principal strain "1" for the second result set

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_plastic_principal_strain('1')
        """
        disp = self.nodal_plastic_principal_strain(component)
        kwargs.setdefault('stitle', '%s Nodal\nPrincipal Strain' % component)
        return self._plot_point_scalars(disp, show_node_numbering=show_node_numbering,
                                        **kwargs)

    @property
    def nodal_plastic_strain_intensity(self) -> np.ndarray:
        """The plastic nodal strain intensity of the current result.

        Equivalent MAPDL command:
        ``PRNSOL, EPPL, PRIN``

        Examples
        --------
        Plastic strain intensity for result 2

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.nodal_plastic_strain_intensity
        array([15488.84357602, 16434.95432337, 15683.2334295 , ...,
                   0.        ,     0.        ,     0.        ])

        Notes
        -----
        The nodal results are averaged across all selected elements.
        Not all nodes will contain valid results (e.g. midside nodes),
        and those nodes will report a zero value.

        Elements that are not selected will not contribute to the
        averaged nodal values, and if a node's attached elements are
        all unselected, the element will report a zero value.
        """
        return self._ndof_rst('EPPL', 'INT')

    def plot_nodal_plastic_strain_intensity(self,
                                            show_node_numbering=False,
                                            **kwargs):
        """Plot the plastic nodal strain intensity of the current result.

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the plastic strain intensity for the second result

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_plastic_strain_intensity()

        Plot off_screen and save a screenshot

        >>> mapdl.post_processing.plot_nodal_plastic_strain_intensity(off_screen=True,
                                                                    screenshot='seqv_00.png')

        Subselect a single result type and plot those strain results
        >>> mapdl.esel('S', 'TYPE', vmin=1)
        >>> mapdl.post_processing.plot_nodal_plastic_strain_intensity()

        """
        scalars = self.nodal_plastic_strain_intensity
        kwargs.setdefault('stitle', 'Plastic Nodal\nStrain Intensity')
        return self._plot_point_scalars(scalars,
                                        show_node_numbering=show_node_numbering,
                                        **kwargs)

    @property
    def nodal_plastic_eqv_strain(self) -> np.ndarray:
        """The plastic nodal equivalent strain of the current result.

        Equivalent MAPDL command:
        ``PRNSOL, EPPL, PRIN``

        Examples
        --------
        Plastic quivalent strain for the current result

        >>> mapdl.post_processing.nodal_plastic_eqv_strain
        array([15488.84357602, 16434.95432337, 15683.2334295 , ...,
                   0.        ,     0.        ,     0.        ])

        Strain from result 2

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.nodal_plastic_eqv_strain
        array([15488.84357602, 16434.95432337, 15683.2334295 , ...,
                   0.        ,     0.        ,     0.        ])

        Notes
        -----
        The nodal results are averaged across all selected elements.
        Not all nodes will contain valid results (e.g. midside nodes),
        and those nodes will report a zero value.

        Elements that are not selected will not contribute to the
        averaged nodal values, and if a node's attached elements are
        all unselected, the element will report a zero value.
        """
        return self._ndof_rst('EPPL', 'EQV')

    def plot_nodal_plastic_eqv_strain(self, show_node_numbering=False, **kwargs):
        """Plot the plastic nodal equivalent strain of the current result.

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the plastic equivalent strain for the second result

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_plastic_eqv_strain()

        Plot off_screen and save a screenshot

        >>> mapdl.post_processing.plot_nodal_plastic_eqv_strain(off_screen=True,
                                                              screenshot='seqv_00.png')

        Subselect a single result type and plot those strain results
        >>> mapdl.esel('S', 'TYPE', vmin=1)
        >>> mapdl.post_processing.plot_nodal_plastic_eqv_strain(smooth_shading=True)

        """
        scalars = self.nodal_plastic_eqv_strain
        kwargs.setdefault('stitle', 'Plastic Nodal\n Equivalent Strain')
        return self._plot_point_scalars(scalars,
                                        show_node_numbering=show_node_numbering,
                                        **kwargs)


###############################################################################


    def nodal_thermal_component_strain(self, component) -> np.ndarray:
        """Thermal nodal component strain

        Equivalent MAPDL command:
        PRNSOL, EPTH, PRIN

        Parameters
        ----------
        component : str, optional
            Component to retrieve.  Must be ``'X'``, ``'Y'``, ``'Z'``,
            ``'XY'``, ``'YZ'``, or ``'XZ'``.

        Examples
        --------
        Thermal component strain in the X direction for the first result

        >>> mapdl.post1()
        >>> mapdl.set(1, 1)
        >>> mapdl.post_processing.nodal_thermal_component_strain('X')
            array([0.60024621, 0.61625265, 0.65081825, ...,
                   0.        , 0.        , 0.        ])

        Corresponding nodes

        >>> mapdl.mesh.nnum_all
        array([   1,    2,    3, ..., 7215, 7216, 7217], dtype=int32)

        Notes
        -----
        This command always returns all nodal rotations regardless of
        if the nodes are selected or not.  Use the ``selected_nodes``
        mask to get the currently selected nodes.
        """
        if isinstance(component, int):
            component = str(component)
        component = check_comp(component, COMPONENT_STRESS_TYPE)
        return self._ndof_rst('EPTH', component)

    def plot_nodal_thermal_component_strain(self, component, show_node_numbering=False,
                                            **kwargs):
        """Plot nodal thermal component strain.

        Parameters
        ----------
        component : str
            Nodal thermal component to plot.  Must be ``'X'``,
            ``'Y'``, ``'Z'``, ``'XY'``, ``'YZ'``, or ``'XZ'``.

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the nodal thermal principal strain "1" for the second result set

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_thermal_component_strain('1')
        """
        disp = self.nodal_thermal_component_strain(component)
        kwargs.setdefault('stitle', '%s Thermal Nodal\nComponent Strain' % component)
        return self._plot_point_scalars(disp, show_node_numbering=show_node_numbering,
                                        **kwargs)

    def nodal_thermal_principal_strain(self, component) -> np.ndarray:
        """Nodal thermal principal thermal strain.

        Equivalent MAPDL commands:
        \*VGET, PARM, NODE, , EPTH, 1

        Parameters
        ----------
        component : str, optional
            Component to retrieve.  Must be ``'1'``, ``'2'``, or
            ``'3'``

        Examples
        --------
        Principal nodal strain in the S1 direction for the first result

        >>> mapdl.post1()
        >>> mapdl.set(1, 1)
        >>> mapdl.post_processing.nodal_thermal_principal_strain('1')
            array([0.60024621, 0.61625265, 0.65081825, ...,
                   0.        , 0.        , 0.        ])

        Corresponding nodes

        >>> mapdl.mesh.nnum_all
        array([   1,    2,    3, ..., 7215, 7216, 7217], dtype=int32)

        Notes
        -----
        This command always returns all nodal rotations regardless of
        if the nodes are selected or not.  Use the ``selected_nodes``
        mask to get the currently selected nodes.
        """
        if isinstance(component, int):
            component = str(component)
        component = check_comp(component, PRINCIPAL_TYPE)
        return self._ndof_rst('EPTH', component)

    def plot_nodal_thermal_principal_strain(self, component,
                                            show_node_numbering=False,
                                            **kwargs):
        """Plot thermal nodal principal strain.

        Parameters
        ----------
        component : str
            Nodal principal strain component to plot.  Must be
            ``'1'``, ``'2'``, or ``'3'``

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the nodal principal strain "1" for the second result set

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_thermal_principal_strain('1')
        """
        disp = self.nodal_thermal_principal_strain(component)
        kwargs.setdefault('stitle', '%s Nodal\nPrincipal Strain' % component)
        return self._plot_point_scalars(disp, show_node_numbering=show_node_numbering,
                                        **kwargs)

    @property
    def nodal_thermal_strain_intensity(self) -> np.ndarray:
        """The thermal nodal strain intensity of the current result.

        Equivalent MAPDL command:
        ``PRNSOL, EPTH, PRIN``

        Examples
        --------
        Thermal strain intensity for result 2

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.nodal_thermal_strain_intensity
        array([15488.84357602, 16434.95432337, 15683.2334295 , ...,
                   0.        ,     0.        ,     0.        ])

        Notes
        -----
        The nodal results are averaged across all selected elements.
        Not all nodes will contain valid results (e.g. midside nodes),
        and those nodes will report a zero value.

        Elements that are not selected will not contribute to the
        averaged nodal values, and if a node's attached elements are
        all unselected, the element will report a zero value.
        """
        return self._ndof_rst('EPTH', 'INT')

    def plot_nodal_thermal_strain_intensity(self,
                                            show_node_numbering=False,
                                            **kwargs):
        """Plot the thermal nodal strain intensity of the current result.

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the thermal strain intensity for the second result

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_thermal_strain_intensity()

        Plot off_screen and save a screenshot

        >>> mapdl.post_processing.plot_nodal_thermal_strain_intensity(off_screen=True,
                                                                    screenshot='seqv_00.png')

        Subselect a single result type and plot those strain results
        >>> mapdl.esel('S', 'TYPE', vmin=1)
        >>> mapdl.post_processing.plot_nodal_thermal_strain_intensity()

        """
        scalars = self.nodal_thermal_strain_intensity
        kwargs.setdefault('stitle', 'Thermal Nodal\nStrain Intensity')
        return self._plot_point_scalars(scalars,
                                        show_node_numbering=show_node_numbering,
                                        **kwargs)

    @property
    def nodal_thermal_eqv_strain(self) -> np.ndarray:
        """The thermal nodal equivalent strain of the current result.

        Equivalent MAPDL command:
        ``PRNSOL, EPTH, PRIN``

        Examples
        --------
        Thermal quivalent strain for the current result

        >>> mapdl.post_processing.nodal_thermal_eqv_strain
        array([15488.84357602, 16434.95432337, 15683.2334295 , ...,
                   0.        ,     0.        ,     0.        ])

        Strain from result 2

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.nodal_thermal_eqv_strain
        array([15488.84357602, 16434.95432337, 15683.2334295 , ...,
                   0.        ,     0.        ,     0.        ])

        Notes
        -----
        The nodal results are averaged across all selected elements.
        Not all nodes will contain valid results (e.g. midside nodes),
        and those nodes will report a zero value.

        Elements that are not selected will not contribute to the
        averaged nodal values, and if a node's attached elements are
        all unselected, the element will report a zero value.
        """
        return self._ndof_rst('EPTH', 'EQV')

    def plot_nodal_thermal_eqv_strain(self, show_node_numbering=False, **kwargs):
        """Plot the thermal nodal equivalent strain of the current result.

        Returns
        --------
        cpos : list
            Camera position from plotter.  Can be reused as an input
            parameter to use the same camera position for future
            plots.

        Examples
        --------
        Plot the thermal equivalent strain for the second result

        >>> mapdl.post1()
        >>> mapdl.set(1, 2)
        >>> mapdl.post_processing.plot_nodal_thermal_eqv_strain()

        Plot off_screen and save a screenshot

        >>> mapdl.post_processing.plot_nodal_thermal_eqv_strain(off_screen=True,
                                                              screenshot='seqv_00.png')

        Subselect a single result type and plot those strain results
        >>> mapdl.esel('S', 'TYPE', vmin=1)
        >>> mapdl.post_processing.plot_nodal_thermal_eqv_strain(smooth_shading=True)

        """
        scalars = self.nodal_thermal_eqv_strain
        kwargs.setdefault('stitle', 'Thermal Nodal\n Equivalent Strain')
        return self._plot_point_scalars(scalars,
                                        show_node_numbering=show_node_numbering,
                                        **kwargs)

