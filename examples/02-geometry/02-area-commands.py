"""
.. _ref_basic-geometry-areas:

Line Commands
-------------
This example shows how you can use PyMAPDL to create basic geometry
using Pythonic PREP7 area commands.

"""

# start MAPDL and enter the pre-processing routine
from ansys.mapdl.core import launch_mapdl

mapdl = launch_mapdl()
mapdl.clear()
mapdl.prep7()
print(mapdl)


###############################################################################
# APDL Command: A
# ~~~~~~~~~~~~~~~
# Create a simple triangle in the XY plane using three keypoints.

k0 = mapdl.k("", 0, 0, 0)
k1 = mapdl.k("", 1, 0, 0)
k2 = mapdl.k("", 0, 1, 0)
a0 = mapdl.a(k0, k1, k2)
mapdl.aplot(show_lines=True, line_width=5, show_bounds=True, cpos='xy')


###############################################################################
# APDL Command: AL
# ~~~~~~~~~~~~~~~~
# Create an area from four lines.
mapdl.clear(); mapdl.prep7()

k0 = mapdl.k("", 0, 0, 0)
k1 = mapdl.k("", 1, 0, 0)
k2 = mapdl.k("", 1, 1, 0)
k3 = mapdl.k("", 0, 1, 0)
l0 = mapdl.l(k0, k1)
l1 = mapdl.l(k1, k2)
l2 = mapdl.l(k2, k3)
l3 = mapdl.l(k3, k0)
anum = mapdl.al(l0, l1, l2, l3)
mapdl.aplot(show_lines=True, line_width=5, show_bounds=True, cpos='xy')


###############################################################################
# APDL Command: ADRAG
# ~~~~~~~~~~~~~~~~~~~
# Generate areas by dragging a line pattern along a path.
#
# Drag a circle between two keypoints to create an area
mapdl.clear(); mapdl.prep7()

k0 = mapdl.k("", 0, 0, 0)
k1 = mapdl.k("", 0, 0, 1)
carc = mapdl.circle(k0, 1, k1, arc=90)
l0 = mapdl.l(k0, k1)
mapdl.adrag(carc[0], nlp1=l0)
mapdl.aplot(show_lines=True, line_width=5, show_bounds=True, smooth_shading=True)


###############################################################################
# APDL Command: BLC4
# ~~~~~~~~~~~~~~~~~~
# Createa a ``0.5 x 0.5`` rectangle starting at ``(0.25, 0.25)``
mapdl.clear(); mapdl.prep7()

anum1 = mapdl.blc4(0.25, 0.25, 0.5, 0.5)
mapdl.aplot(show_lines=True, line_width=5, show_bounds=True, cpos='xy')


###############################################################################
# APDL Command: ASBA
# ~~~~~~~~~~~~~~~~~~
# Subtract a ``0.5 x 0.5`` rectangle from a ``1 x 1`` rectangle.
mapdl.clear(); mapdl.prep7()

anum0 = mapdl.blc4(0, 0, 1, 1)
anum1 = mapdl.blc4(0.25, 0.25, 0.5, 0.5)
aout = mapdl.asba(anum0, anum1)
mapdl.aplot(show_lines=True, line_width=5, show_bounds=True, cpos='xy')


###############################################################################
# APDL Command: CYL4
# ~~~~~~~~~~~~~~~~~~
# Create a half arc centered at the origin with an outer radius
# of 2 and an inner radius of 1.
#
# Note that the ``depth`` keyword argument is unset, which will
# generate an area rather than a volume.  Setting depth to a value
# greater than 0 will generate a volume.
mapdl.clear(); mapdl.prep7()

anum = mapdl.cyl4(xcenter=0, ycenter=0, rad1=1, theta1=0, rad2=2, theta2=180)
mapdl.aplot(show_lines=True, line_width=5, show_bounds=True, cpos='xy')


###############################################################################
# Area IDs
# ~~~~~~~~
# Return an array of the area IDs
anum = mapdl.geometry.anum
anum


###############################################################################
# Area Geometry
# ~~~~~~~~~~~~~
# Get the VTK ``PolyData`` containing lines.  This VTK mesh can be
# saved or plotted.  For more details, visit https://docs.pyvista.com
#
# Note that this is a method so you can select the quality of the
# areas (mesh density), and if you would like a merged output or
# individual meshes.
areas = mapdl.geometry.areas(quality=3)
areas


###############################################################################
# Merged area
area = mapdl.geometry.areas(quality=3, merge=True)
area

# optionally save the area, or plot it
# area.save('mesh.vtk')
# area.plot()



###############################################################################
# APDL Command: APLOT
# ~~~~~~~~~~~~~~~~~~~
# This method uses VTK and pyvista to generate a dynamic 3D plot.
#
# There are a variety of plotting options available for all the common
# plotting methods.  Here, we enable the bounds and show the lines of
# the plot while increasing the plot quality with the ``quality``
# parameter.
#
# Note that the `cpos` keyword argument can be used to describe the
# camera direction from the following:
#
# - 'iso' - Isometric view
# - 'xy' - XY Plane view
# - 'xz' - XZ Plane view
# - 'yx' - YX Plane view
# - 'yz' - YZ Plane view
# - 'zx' - ZX Plane view
# - 'zy' - ZY Plane view

mapdl.aplot(quality=1,
            show_bounds=True,
            cpos='iso',
            show_lines=True)
