"""Module to support MAPDL CAD geometry"""
import re

import numpy as np
import pyvista as pv

from ansys.mapdl.core.misc import supress_logging, run_as_prep7


def merge_polydata(items):
    """Merge list of polydata or unstructured grids"""

    # lazy import here for faster module loading
    try:
        from pyvista._vtk import vtkAppendPolyData
    except:
        from vtk import vtkAppendPolyData

    afilter = vtkAppendPolyData()
    for item in items:
        afilter.AddInputData(item)
        afilter.Update()

    return pv.wrap(afilter.GetOutput())


def get_elements_per_area(resp):
    """Get the number of elements meshed for each area given the response
    from ``AMESH``.

    GENERATE NODES AND ELEMENTS   IN  ALL  SELECTED AREAS
        ** AREA     1 MESHED WITH      64 QUADRILATERALS,        0 TRIANGLES **
        ** AREA     2 MESHED WITH      64 QUADRILATERALS,        0 TRIANGLES **
        ** AREA     3 MESHED WITH      64 QUADRILATERALS,        0 TRIANGLES **
        ** AREA     4 MESHED WITH      64 QUADRILATERALS,        0 TRIANGLES **
        ** AREA     5 MESHED WITH      64 QUADRILATERALS,        0 TRIANGLES **
        ** AREA     6 MESHED WITH      64 QUADRILATERALS,        0 TRIANGLES **
        ** AREA     7 MESHED WITH      64 QUADRILATERALS,        0 TRIANGLES **
        ** AREA     8 MESHED WITH      64 QUADRILATERALS,        0 TRIANGLES **
        ** AREA     9 MESHED WITH      64 QUADRILATERALS,        0 TRIANGLES **
        ** AREA    10 MESHED WITH      64 QUADRILATERALS,        0 TRIANGLES **
        ** AREA    11 MESHED WITH      64 QUADRILATERALS,        0 TRIANGLES **
        ** AREA    12 MESHED WITH      64 QUADRILATERALS,        0 TRIANGLES **

     NUMBER OF AREAS MESHED     =         12
     MAXIMUM NODE NUMBER        =        772
     MAXIMUM ELEMENT NUMBER     =        768

    Returns
    -------
    list
        List of tuples, each containing the area number and number of
        elements per area.

    """
    # MAPDL changed their output at some point.  Check for both output types.
    reg = re.compile(r'Meshing of area (\d*) completed \*\* (\d*) elements')
    groups = reg.findall(resp)
    if groups:
        groups = [[int(anum), int(nelem)] for anum, nelem in groups]
    else:
        reg = re.compile(r'AREA\s*(\d*).*?(\d*)\s*QUADRILATERALS,\s*(\d*) TRIANGLES')
        groups = reg.findall(resp)
        groups = [(int(anum), int(nquad) + int(ntri)) for anum, nquad, ntri in groups]

    return groups


class Geometry():
    """Pythonic representation of MAPDL CAD geometry

    Contains advanced methods to extend geometry building and
    selection within MAPDL.

    """

    def __init__(self, mapdl):
        from ansys.mapdl.core.mapdl import _MapdlCore
        if not isinstance(mapdl, _MapdlCore):
            raise TypeError('Must be initialized using a MAPDL class')

        self._mapdl = mapdl
        self._keypoints_cache = None
        self._lines_cache = None
        self._log = self._mapdl._log

    def _set_log_level(self, level):
        return self._mapdl.set_log_level(level)

    def _load_iges(self):
        """Loads the iges file from MAPDL as a pyiges class"""
        # Lazy import here for speed and stability
        # possible to exclude this import in the future
        try:
            from pyiges import Iges
        except ImportError:
            raise ImportError('Please install pyiges to use this feature with:\n'
                              'pip install pyiges')
        return Iges(self._mapdl._generate_iges())

    def _reset_cache(self):
        self._keypoints_cache = None
        self._lines_cache = None

    @property
    def _keypoints(self):
        """Returns keypoints cache"""
        if self._keypoints_cache is None:
            self._keypoints_cache = self._load_keypoints()
        return self._keypoints_cache

    @property
    def keypoints(self):
        """Keypoint coordinates"""
        return np.asarray(self._keypoints.points)

    @property
    def _lines(self):
        """Returns lines cache"""
        if self._lines_cache is None:
            self._lines_cache = self._load_lines()
        return self._lines_cache

    @property
    def lines(self):
        """Active lines as a pyvista.PolyData"""
        return self._lines

    def areas(self, quality=4, merge=False):
        """List of areas from MAPDL represented as ``pyvista.PolyData``.

        Parameters
        ----------
        quality : int, optional
            quality of the mesh to display.  Varies between 1 (worst)
            to 10 (best).

        merge : bool, optional
            Option to merge areas into a single mesh. Default
            ``False`` to return a list of areas.  When ``True``,
            output will be a single mesh.

        Returns
        -------
        areas : list, pyvista.UnstructuredGrid
            List of ``pyvista.UnstructuredGrid`` meshes representing
            the active surface areas selected by ``ASEL``.  If
            ``merge=True``, areas are returned as a single merged
            UnstructuredGrid.

        Examples
        --------
        Return a list of areas as indiviudal grids

        >>> areas = mapdl.areas(quality=3)
        >>> areab
        [UnstructuredGrid (0x7f14add95040)
          N Cells:	12
          N Points:	20
          X Bounds:	-2.000e+00, 2.000e+00
          Y Bounds:	0.000e+00, 1.974e+00
          Z Bounds:	0.000e+00, 0.000e+00
          N Arrays:	4,
        UnstructuredGrid (0x7f14add95ca0)
          N Cells:	12
          N Points:	20
          X Bounds:	-2.000e+00, 2.000e+00
          Y Bounds:	0.000e+00, 1.974e+00
          Z Bounds:	5.500e-01, 5.500e-01
          N Arrays:	4,
        ...

        Return a single merged mesh.

        >>> area_mesh = mapdl.areas(quality=3)
        >>> area_mesh
        UnstructuredGrid (0x7f14add95ca0)
          N Cells:	24
          N Points:	30
          X Bounds:	-2.000e+00, 2.000e+00
          Y Bounds:	0.000e+00, 1.974e+00
          Z Bounds:	5.500e-01, 5.500e-01
          N Arrays:	4


        """
        quality = quality(int)
        if quality > 10:
            raise ValueError('``quality`` parameter must be a value between 0 and 10')
        surf = self.generate_surface(11 - quality)
        if merge:
            return surf

        entity_num = surf['entity_num']
        areas = []
        anums = np.unique(entity_num)
        for anum in anums:
            areas.append(surf.extract_cells(entity_num == anum))

        return areas

    @supress_logging
    @run_as_prep7
    def generate_surface(self, density=4, amin=None, amax=None, ninc=None):
        """Generate an all-triangular surface of the active surfaces.

        Parameters
        ----------
        density : int, optional
            APDL smart sizing option.  Ranges from 1 (worst) to 10
            (best).

        amin : int, optional
            Minimum APDL numbered area to select.  See
            ``mapdl.anum`` for available areas.

        amax : int, optional
            Maximum APDL numbered area to select.  See
            ``mapdl.anum`` for available areas.

        ninc : int, optional
            Steps to between amin and amax.
        """
        # store initially selected areas and elements
        with self._mapdl.chain_commands:
            self._mapdl.cm('__tmp_elem__', 'ELEM')
            self._mapdl.cm('__tmp_area__', 'AREA')
        orig_anum = self.anum

        # reselect from existing selection to mimic APDL behavior
        if amin or amax:
            if amax is None:
                amax = amin
            else:
                amax = ''

            if amin is None:  # amax is non-zero
                amin = 1

            if ninc is None:
                ninc = ''

            self._mapdl.asel('R', 'AREA', vmin=amin, vmax=amax, vinc=ninc)

        # duplicate areas to avoid affecting existing areas
        a_num = int(self._mapdl.get(entity='AREA', item1='NUM', it1num='MAXD'))
        with self._mapdl.chain_commands:
            self._mapdl.numstr('AREA', a_num)
            self._mapdl.agen(2, 'ALL', noelem=1)
        a_max = int(self._mapdl.get(entity='AREA', item1='NUM', it1num='MAXD'))

        with self._mapdl.chain_commands:
            self._mapdl.asel('S', 'AREA', vmin=a_num + 1, vmax=a_max)
            self._mapdl.aatt()  # necessary to reset element/area meshing association

        # create a temporary etype
        etype_max = int(self._mapdl.get(entity='ETYP', item1='NUM', it1num='MAX'))
        etype_old = self._mapdl.parameters.type
        etype_tmp = etype_max + 1

        old_routine = self._mapdl.parameters.routine

        with self._mapdl.chain_commands:
            self._mapdl.et(etype_tmp, 'MESH200', 6)
            self._mapdl.shpp('off')
            self._mapdl.smrtsize(density)
            self._mapdl.type(etype_tmp)

            if old_routine != 'PREP7':
                self._mapdl.prep7()

        # Mesh and get the number of elements per area
        resp = self._mapdl.amesh('all')
        groups = get_elements_per_area(resp)

        self._mapdl.esla('S')
        grid = self._mapdl.mesh._grid.linear_copy()
        pd = pv.PolyData(grid.points, grid.cells, n_faces=grid.n_cells)

        pd['ansys_node_num'] = grid['ansys_node_num']
        pd['vtkOriginalPointIds'] = grid['vtkOriginalPointIds']
        # pd.clean(inplace=True)  # OPTIONAL

        # delete all temporary meshes and clean up settings
        with self._mapdl.chain_commands:
            self._mapdl.aclear('ALL')
            self._mapdl.adele('ALL', kswp=1)
            self._mapdl.numstr('AREA', 1)
            self._mapdl.type(etype_old)
            self._mapdl.etdele(etype_tmp)
            self._mapdl.shpp('ON')
            self._mapdl.smrtsize('OFF')
            self._mapdl.cmsel('S', '__tmp_area__', 'AREA')
            self._mapdl.cmsel('S', '__tmp_elem__', 'ELEM')

        # ensure the number of groups matches the number of areas
        if len(groups) != len(orig_anum):
            groups = None

        # store the area number used for each element
        entity_num = np.empty(grid.n_cells, dtype=np.int32)
        if grid and groups:
            # add anum info
            i = 0
            for index, (anum, nelem) in enumerate(groups):
                # have to use original area numbering here as the
                # duplicated areas numbers are inaccurate
                entity_num[i:i+nelem] = orig_anum[index]
                i += nelem
        else:
            entity_num[:] = 0

        pd['entity_num'] = entity_num
        return pd

    @property
    def n_volu(self):
        """Number of volumes currently selected

        Examples
        --------
        >>> mapdl.n_area
        1
        """
        return self._item_count('VOLU')

    @property
    def n_area(self):
        """Number of areas currently selected

        Examples
        --------
        >>> mapdl.n_area
        1
        """
        return self._item_count('AREA')

    @property
    def n_line(self):
        """Number of lines currently selected

        Examples
        --------
        >>> mapdl.n_line
        1
        """
        return self._item_count('LINE')

    @property
    def n_keypoint(self):
        """Number of keypoints currently selected

        Examples
        --------
        >>> mapdl.n_keypoint
        1
        """
        return self._item_count('KP')

    @supress_logging
    def _item_count(self, entity):
        """Return item count for a given entity"""
        return int(self._mapdl.get(entity=entity, item1='COUNT'))

    @property
    def knum(self):
        """Array of keypoint numbers.

        Examples
        --------
        >>> mapdl.block(0, 1, 0, 1, 0, 1)
        >>> mapdl.knum
        array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int32)
        """
        return self._mapdl.get_array('KP', item1='KLIST').astype(np.int32)    

    @property
    def lnum(self):
        """Array of line numbers.

        Examples
        --------
        >>> mapdl.block(0, 1, 0, 1, 0, 1)
        >>> mapdl.lnum
        array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12], dtype=int32)
        """
        # this (weirdly) sometimes fails
        for _ in range(5):
            lnum = self._mapdl.get_array('LINES', item1='LLIST')
            if lnum.size == self.n_line:
                break
        return lnum.astype(np.int32)

    @property
    def anum(self):
        """Array of area numbers.
        Examples
        --------
        >>> mapdl.block(0, 1, 0, 1, 0, 1)
        >>> mapdl.anum
        array([1, 2, 3, 4, 5, 6], dtype=int32)
        """
        return self._mapdl.get_array('AREA', item1='ALIST').astype(np.int32)

    @property
    def vnum(self):
        """Array of volume numbers.

        Examples
        --------
        >>> mapdl.block(0, 1, 0, 1, 0, 1)
        >>> mapdl.vnum
        array([1], dtype=int32)
        """
        return self._mapdl.get_array('VOLU', item1='VLIST').astype(np.int32)

    @supress_logging
    def _load_lines(self):
        """Load lines from MAPDL using IGES"""
        # ignore volumes
        with self._mapdl.chain_commands:
            self._mapdl.cm('__tmp_volu__', 'VOLU')
            self._mapdl.cm('__tmp_line__', 'LINE')
            self._mapdl.cm('__tmp_area__', 'AREA')
            self._mapdl.cm('__tmp_keyp__', 'KP')
            self._mapdl.ksel('ALL')
            self._mapdl.lsel('ALL')
            self._mapdl.asel('ALL')
            self._mapdl.vsel('NONE')

        iges = self._load_iges()

        with self._mapdl.chain_commands:
            self._mapdl.cmsel('S', '__tmp_volu__', 'VOLU')
            self._mapdl.cmsel('S', '__tmp_area__', 'AREA')
            self._mapdl.cmsel('S', '__tmp_line__', 'LINE')
            self._mapdl.cmsel('S', '__tmp_keyp__', 'KP')

        selected_lnum = self.lnum
        lines = []
        entity_nums = []
        for bspline in iges.bsplines():
            # allow only 10001 as others appear to be construction entities
            if bspline.d['status_number'] in [1, 10001]:
                entity_num = int(bspline.d['entity_subs_num'])
                if entity_num not in entity_nums and entity_num in selected_lnum:
                    entity_nums.append(entity_num)
                    line = bspline.to_vtk()
                    line.cell_arrays['entity_num'] = entity_num
                    lines.append(line)

        entities = iges.lines() + iges.circular_arcs()
        for line in entities:
            if line.d['status_number'] == 1:
                entity_num = int(line.d['entity_subs_num'])
                if entity_num not in entity_nums and entity_num in selected_lnum:
                    entity_nums.append(entity_num)
                    line = line.to_vtk(resolution=100)
                    line.cell_arrays['entity_num'] = entity_num
                    lines.append(line)

        if lines:
            lines = merge_polydata(lines)
            lines['entity_num'] = lines['entity_num'].astype(np.int32)
        else:
            lines = pv.PolyData()

        return lines

    def _load_keypoints(self):
        """Load keypoints from MAPDL using IGES"""
        # write only keypoints
        with self._mapdl.chain_commands:
            self._mapdl.cm('__tmp_volu__', 'VOLU')
            self._mapdl.cm('__tmp_area__', 'AREA')
            self._mapdl.cm('__tmp_line__', 'LINE')
            self._mapdl.vsel('NONE')
            self._mapdl.asel('NONE')
            self._mapdl.lsel('NONE')

        iges = self._load_iges()

        with self._mapdl.chain_commands:
            self._mapdl.cmsel('S', '__tmp_volu__', 'VOLU')
            self._mapdl.cmsel('S', '__tmp_area__', 'AREA')
            self._mapdl.cmsel('S', '__tmp_line__', 'LINE')

        keypoints = []
        kp_num = []
        for kp in iges.points():
            keypoints.append([kp.x, kp.y, kp.z])
            kp_num.append(int(kp.d['entity_subs_num']))

        # self._kp_num = np.array(self._kp_num)
        keypoints_pd = pv.PolyData(keypoints)
        keypoints_pd['entity_num'] = kp_num
        return keypoints_pd

    def __str__(self):
        """Current geometry info"""
        info = 'MAPDL Selected Geometry\n'
        info += 'Keypoints:  %d\n' % self.n_keypoint
        info += 'Lines:      %d\n' % self.n_line
        info += 'Areas:      %d\n' % self.n_area
        info += 'Volumes:    %d\n' % self.n_volu
        return info
