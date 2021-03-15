"""
.. _ref_basic-geometry-keypoints:

Keypoints
---------
This example shows how you can use PyMAPDL to create basic geometry
using Pythonic PREP7 commands.

This section is focused on creating keypoints.

"""

import numpy as np
from ansys.mapdl.core import launch_mapdl

# start MAPDL and enter the pre-processing routine
mapdl = launch_mapdl()
mapdl.clear()
mapdl.prep7()
print(mapdl)


###############################################################################
# APDL Command: K
# ~~~~~~~~~~~~~~~
# Create a single keypoint at ``[0, 0, 0]``.  Note that the first
# value is an empty string to allow MAPDL to pick the keypoint number.
k0 = mapdl.k('', 0, 0, 0)
print(k0)


###############################################################################
# Create keypoint at (10, 11, 12) while specifying the keypoint number.
k1 = mapdl.k(2, 1, 0, 0)
print(k1)


###############################################################################
# APDL Command: KBETW
# ~~~~~~~~~~~~~~~~~~~
# Create keypoint between two keypoints
k2 = mapdl.kbetw(kp1=k0, kp2=k1)
print(k2)


###############################################################################
# APDL Command: KCENTER
# ~~~~~~~~~~~~~~~~~~~~~
# Create keypoint at the center of a circular arc defined by three locations.
# Note that we first clear mapdl before generating this geometry
mapdl.clear()
mapdl.prep7()
k0 = mapdl.k("", 0, 1, 0)
k1 = mapdl.k("", 1, 0, 0)
k2 = mapdl.k("", 0, -1, 0)
k3 = mapdl.kcenter('KP', k0, k1, k2)
print([k0, k1, k2, k3])


###############################################################################
# Keypoint IDs
# ~~~~~~~~~~~~
# Return an array of the keypoint IDs
# Note that this matches the array ``[k0, k1, k2, k3]`` (due to sorting)
knum = mapdl.geometry.knum
knum

###############################################################################
# Keypoint Coordinates
# ~~~~~~~~~~~~~~~~~~~~
# Return an array of the keypoint locations
kloc = mapdl.geometry.keypoints
kloc


###############################################################################
# APDL Command: KDIST
# ~~~~~~~~~~~~~~~~~~~
# Calculate the distance between two keypoints.  Note that you could
# compute this yourself from ``kloc``
dist = mapdl.kdist(k0, k1)
dist


###############################################################################
# Keypoint Selection
# ~~~~~~~~~~~~~~~~~~
# There are two approaches for selecting keypoints, the old "legacy"
# style and the new style.  The old style is valuable for those who
# are comfortable with the existing MAPDL commands, and new style is
# useful for selecting keypoints in a pythonic manner.
#
# This example generates a series of random keypoints and selects them
mapdl.clear(); mapdl.prep7()

# create 20 random keypoints
for _ in range(20):
    mapdl.k('', *np.random.random(3))

# Print the keypoint numbers
print(mapdl.geometry.knum)


###############################################################################
# Select every other keypoint with the old style command.
mapdl.ksel('S', 'KP', '', 1, 20, 2)
print(mapdl.geometry.knum)


###############################################################################
# Select every other keypoint with the new style command.
#
# Note that the item IDs are 1 based in MAPDL, while Python ranges are
# 0 based.
mapdl.geometry.keypoint_select(range(1, 21, 2))
print(mapdl.geometry.knum)


###############################################################################
# Select keypoints from a list
#
# Note that you can ``return_selected`` if you want to see what you
# have selected.  This is helpful when reselecting from existing
# areas.
#
# Note that you could use a numpy array here as well.
items = mapdl.geometry.keypoint_select([1, 5, 10, 20], return_selected=True)
print(items)


###############################################################################
# APDL Command: KPLOT
# ~~~~~~~~~~~~~~~~~~~
# Plot the keypoints while displaying the keypoint numbers
#
# There are a variety of plotting options available for all the common
# plotting methods.
mapdl.kplot(show_keypoint_numbering=True,
            background='black',
            show_bounds=True,
            font_size=26)
