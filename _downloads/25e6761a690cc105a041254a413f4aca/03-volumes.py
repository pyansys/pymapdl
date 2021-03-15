"""
.. _ref_basic-geometry-volumes:

Volumes
-------
This example shows how you can use PyMAPDL to create basic geometry
using Pythonic PREP7 volume commands.

"""

import numpy as np
from ansys.mapdl.core import launch_mapdl

# start MAPDL and enter the pre-processing routine
mapdl = launch_mapdl()
mapdl.clear()
mapdl.prep7()
print(mapdl)


###############################################################################
# APDL Command: V
# ~~~~~~~~~~~~~~~
# Define a volume through keypoints.

# Create a simple cube volume.

k0 = mapdl.k("", 0, 0, 0)
k1 = mapdl.k("", 1, 0, 0)
k2 = mapdl.k("", 1, 1, 0)
k3 = mapdl.k("", 0, 1, 0)
k4 = mapdl.k("", 0, 0, 1)
k5 = mapdl.k("", 1, 0, 1)
k6 = mapdl.k("", 1, 1, 1)
k7 = mapdl.k("", 0, 1, 1)
v0 = mapdl.v(k0, k1, k2, k3, k4, k5, k6, k7)
mapdl.vplot(show_lines=True)


###############################################################################
# Create a triangular prism
mapdl.clear(); mapdl.prep7()

k0 = mapdl.k("", 0, 0, 0)
k1 = mapdl.k("", 1, 0, 0)
k2 = mapdl.k("", 1, 1, 0)
k3 = mapdl.k("", 0, 1, 0)
k4 = mapdl.k("", 0, 0, 1)
k5 = mapdl.k("", 1, 0, 1)
k6 = mapdl.k("", 1, 1, 1)
k7 = mapdl.k("", 0, 1, 1)
v1 = mapdl.v(k0, k1, k2, k2, k4, k5, k6, k6)
mapdl.vplot(show_lines=True)


###############################################################################
# Create a triangular prism
mapdl.clear(); mapdl.prep7()

k0 = mapdl.k("", 0, 0, 0)
k1 = mapdl.k("", 1, 0, 0)
k2 = mapdl.k("", 1, 1, 0)
k3 = mapdl.k("", 0, 0, 1)
v0 = mapdl.v(k0, k1, k2, k2, k3, k3, k3, k3)
mapdl.vplot(show_lines=True)


###############################################################################
# APDL Command: VA
# ~~~~~~~~~~~~~~~~
# Generate a volume bounded by existing areas.
#
# Create a simple tetrahedral bounded by 4 areas

mapdl.clear(); mapdl.prep7()
k0 = mapdl.k('', -1, 0, 0)
k1 = mapdl.k('', 1, 0,  0)
k2 = mapdl.k('', 1, 1, 0)
k3 = mapdl.k('', 1, 0.5, 1)

# create faces
a0 = mapdl.a(k0, k1, k2)
a1 = mapdl.a(k0, k1, k3)
a2 = mapdl.a(k1, k2, k3)
a3 = mapdl.a(k0, k2, k3)

# generate and plot the volume
vnum = mapdl.va(a0, a1, a2, a3)
mapdl.aplot(show_lines=True, show_bounds=True)


###############################################################################
# APDL Command: VDRAG
# ~~~~~~~~~~~~~~~~~~~
# Generate volumes by dragging an area pattern along a path.
#
# Create a square with a hole in it and drag it along an arc.
mapdl.clear(); mapdl.prep7()

# create a square with a hole in it.
anum0 = mapdl.blc4(0, 0, 1, 1)
anum1 = mapdl.blc4(0.25, 0.25, 0.5, 0.5)
aout = mapdl.asba(anum0, anum1)

k0 = mapdl.k("", 0, 0, 0)
k1 = mapdl.k("", 1, 0, 1)
k2 = mapdl.k("", 1, 0, 0)
l0 = mapdl.larc(k0, k1, k2, 2)
mapdl.vdrag(aout, nlp1=l0)
mapdl.vplot(show_lines=True, quality=5)


###############################################################################
# APDL Command: VEXT
# ~~~~~~~~~~~~~~~~~~
# Generate additional volumes by extruding areas.
#
# Create a basic cylinder by extruding a circle.
mapdl.clear(); mapdl.prep7()

# first, create an area from a circle
k0 = mapdl.k("", 0, 0, 0)
k1 = mapdl.k("", 0, 0, 1)
k2 = mapdl.k("", 0, 0, 0.5)
carc0 = mapdl.circle(k0, 1, k1)
a0 = mapdl.al(*carc0)

# next, extrude it and plot it
mapdl.vext(a0, dz=4)
mapdl.vplot(show_lines=True, quality=5)


###############################################################################
# Create a tapered cylinder by using the ``rx`` and ``ry`` parameters.
mapdl.vdele('all')
mapdl.vext(a0, dz=4, rx=0.3, ry=0.3, rz=1)
mapdl.vplot(show_lines=True, quality=5)


###############################################################################
# APDL Command: VROTAT
# ~~~~~~~~~~~~~~~~~~~~
# Generate cylindrical volumes by rotating an area pattern about an
# axis.
#
# Rotate a circle about the Z axis to create a hoop
mapdl.clear(); mapdl.prep7()

# first, create an area from a circle
hoop_radius = 10
hoop_thickness = 0.5
k0 = mapdl.k("", hoop_radius, 0, 0)
k1 = mapdl.k("", hoop_radius, 1, 0)
k2 = mapdl.k("", hoop_radius, 0, hoop_thickness)
carc0 = mapdl.circle(k0, 1, k1)
a0 = mapdl.al(*carc0)

# define a Z-axis
k_axis0 = mapdl.k("", 0, 0, 0)
k_axis1 = mapdl.k("", 0, 0, 1)

# Rotate about the Z-axis.  By default it will rotate all 360 degrees
mapdl.vrotat(a0, pax1=k_axis0, pax2=k_axis1)
mapdl.vplot(show_lines=True, quality=5)


###############################################################################
# APDL Command: VSYMM
# ~~~~~~~~~~~~~~~~~~~~
# Generate volumes from a volume pattern by symmetry reflection.
#
# Create four blocks by reflecting a single block across the YZ and
# then XZ planes.
mapdl.clear(); mapdl.prep7()

vnum = mapdl.blc4(1, 1, 1, 1, depth=1)
mapdl.vsymm('X', vnum)
mapdl.vsymm('Y', 'ALL')

mapdl.vplot(show_lines=True, show_bounds=True)


###############################################################################
# Volume IDs
# ~~~~~~~~~~
# Return an array of the volume numbers.
vnum = mapdl.geometry.vnum
vnum


###############################################################################
# Volume Geometry
# ~~~~~~~~~~~~~~~
# Volume geometry can be accessed from the ``geometry.areas`` method.
volume_mesh = mapdl.geometry.areas(merge=True)
volume_mesh


###############################################################################
# Volume Selection
# ~~~~~~~~~~~~~~~~
# There are two approaches for selecting volumes, the old "legacy"
# style and the new style.  The old style is valuable for those who
# are comfortable with the existing MAPDL commands, and new style is
# useful for selecting volumes in a pythonic manner.
#
# This example generates a series of random volumes and selects them
mapdl.clear(); mapdl.prep7()


def generate_random_volume():
    start_x, start_y, height, width, depth = np.random.random(5)
    mapdl.blc4(start_x*10, start_y*10, height, width, depth)

# create 20 random volumes
for _ in range(20):
    generate_random_volume()

# Print the volume numbers
print(mapdl.geometry.vnum)


###############################################################################
# Select every other volume with the old style command.
mapdl.vsel('S', 'VOLU', '', 1, 20, 2)
print(mapdl.geometry.vnum)


###############################################################################
# Select every other volume with the new style command.
#
# Note that the item IDs are 1 based in MAPDL, while Python ranges are
# 0 based.
mapdl.geometry.volume_select(range(1, 21, 2))
print(mapdl.geometry.vnum)


###############################################################################
# Select volumes from a list
#
# Note that you can ``return_selected`` if you want to see what you
# have selected.  This is helpful when reselecting from existing
# areas.
#
# Note that you could use a numpy array here as well.
items = mapdl.geometry.volume_select([1, 5, 10, 20], return_selected=True)
print(items)


###############################################################################
# APDL Command: VPLOT
# ~~~~~~~~~~~~~~~~~~~
# Plot colored volumes while displaying the keypoint numbers.
#
# There are a variety of plotting options available for all the common
# plotting methods.

mapdl.clear(); mapdl.prep7()

# Create basic demo geometry
mapdl.cyl4(xcenter=0, ycenter=0, rad1=1, theta1=0, rad2=2, depth=1)
mapdl.vsymm('Y', 'ALL')

# Plot while showing bounds and disabling extra line plotting.
mapdl.vplot(show_bounds=True,
            show_lines=False,
            quality=5)
