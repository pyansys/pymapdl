"""
.. _ref_basic-geometry-lines:

Lines
-----
This example shows how you can use PyMAPDL to create basic geometry
using Pythonic PREP7 line commands.

"""

import numpy as np
from ansys.mapdl.core import launch_mapdl

# start MAPDL and enter the pre-processing routine
mapdl = launch_mapdl()
mapdl.clear()
mapdl.prep7()
print(mapdl)


###############################################################################
# APDL Command: L
# ~~~~~~~~~~~~~~~
#
# Create a line between the two keypoints ``(0, 0, 0)`` and ``(1, 0, 0)``

k0 = mapdl.k("", 0, 0, 0)
k1 = mapdl.k("", 1, 0, 0)
lnum = mapdl.l(k0, k1)
lnum


###############################################################################
# APDL Command: LARC
# ~~~~~~~~~~~~~~~~~~
# Create a circular arc that travels between ``(0, 0, 0)`` and
# ``(1, 1, 0)`` with a radius of curvature of 2.  Plot the result.
mapdl.clear(); mapdl.prep7()

k0 = mapdl.k("", 0, 0, 0)
k1 = mapdl.k("", 1, 1, 0)
k2 = mapdl.k("", 0, 1, 0)
lnum = mapdl.larc(k0, k1, k2, 2)
mapdl.lplot(background='w', color='k', line_width=5, cpos='xy')


###############################################################################
# APDL Command: L2ANG
# ~~~~~~~~~~~~~~~~~~~
# Create two circles and join them with a line.  Plot the result.
mapdl.clear(); mapdl.prep7()

k0 = mapdl.k("", 0, 0, 0)
k1 = mapdl.k("", 0, 0, 1)
k2 = mapdl.k("", 0, 0, 0.5)
carc0 = mapdl.circle(k0, 1, k1)
carc1 = mapdl.circle(k2, 1, k1)
lnum = mapdl.l2ang(carc0[0], carc1[0], 90, 90)
mapdl.lplot(background='w', color='k', line_width=5)


###############################################################################
# APDL Command: L2TAN
# ~~~~~~~~~~~~~~~~~~~
# Create two circular arcs and connect them with a spline.  Plot the result.
mapdl.clear(); mapdl.prep7()

k0 = mapdl.k("", 0, 0, 0)
k1 = mapdl.k("", 0, 0, 1)
k2 = mapdl.k("", -1.5, 1.5, 0)
k3 = mapdl.k("", -1.5, 1.5, 1)
carc0 = mapdl.circle(k0, 1, k1, arc=90)
carc1 = mapdl.circle(k2, 1, k3, arc=90)
lnum = mapdl.l2tan(1, 2)
mapdl.lplot(background='w', color='k', line_width=5, cpos='xy')


###############################################################################
# APDL Command: LANG
# ~~~~~~~~~~~~~~~~~~
# Create a line from a line from ``(0, 0, 0)`` to ``(1, 0, 0)`` to a
# keypoint at ``(1, 1, 1)`` at an angle of 60 degrees.  Plot the result.
mapdl.clear(); mapdl.prep7()

k0 = mapdl.k("", 0, 0, 0)
k1 = mapdl.k("", 1, 0, 0)
lnum = mapdl.l(k0, k1)
k2 = mapdl.k("", 1, 1, 0)
lnum = mapdl.lang(lnum, k2, 60)
mapdl.lplot(background='w', color='k', line_width=5, cpos='xy')


###############################################################################
# APDL Command: LAREA
# ~~~~~~~~~~~~~~~~~~~
# Generate a line on a square between its two corners.
mapdl.clear(); mapdl.prep7()

k0 = mapdl.k("", 0, 0, 0)
k1 = mapdl.k("", 1, 0, 0)
k2 = mapdl.k("", 1, 1, 0)
k3 = mapdl.k("", 0, 1, 0)
a0 = mapdl.a(k0, k1, k2, k3)
lnum = mapdl.larea(k0, k2, a0)
mapdl.lplot(background='w', color='k', line_width=5, cpos='xy')


###############################################################################
# APDL Command: LCOMB
# ~~~~~~~~~~~~~~~~~~~
# Create two lines and combine them.

k0 = mapdl.k("", 0, 0, 0)
k1 = mapdl.k("", 1, 0, 0)
k2 = mapdl.k("", 2, 0, 0)
l0 = mapdl.l(k0, k1)
l1 = mapdl.l(k0, k2)
lout = mapdl.lcomb(l0, l1)


###############################################################################
# APDL Command: LDIV
# ~~~~~~~~~~~~~~~~~~
# Create a single line and divide it into 5 pieces.
mapdl.clear(); mapdl.prep7()

k0 = mapdl.k("", 0, 0, 0)
k1 = mapdl.k("", 1, 0, 0)
l0 = mapdl.l(k0, k1)
mapdl.ldiv(l0, ndiv=5)
mapdl.lplot(background='w', color='k', line_width=5, cpos='xy')


###############################################################################
# APDL Command: LEXTND
# ~~~~~~~~~~~~~~~~~~~~
# Create a circular arc and extend it at one of its keypoints.
mapdl.clear(); mapdl.prep7()

k0 = mapdl.k("", 0, 0, 0)
k1 = mapdl.k("", 0, 0, 1)
carcs = mapdl.circle(k0, 1, k1, arc=90)
lnum = mapdl.lextnd(carcs[0], 3, 1)
mapdl.lplot(background='w', color='k', line_width=5, cpos='xy')


###############################################################################
# APDL Command: LFILLT
# ~~~~~~~~~~~~~~~~~~~~
# Create two intersecting lines at a right angle and add a
# fillet between them.
mapdl.clear(); mapdl.prep7()

k0 = mapdl.k("", 0, 0, 0)
k1 = mapdl.k("", 0, 1, 0)
k2 = mapdl.k("", 1, 0, 0)
l0 = mapdl.l(k0, k1)
l1 = mapdl.l(k0, k2)
lnum = mapdl.lfillt(l0, l1, 0.25)
mapdl.lplot(background='w', color='k', line_width=5, cpos='xy')


###############################################################################
# APDL Command: LTAN
# ~~~~~~~~~~~~~~~~~~
# Create a circular arc and generate a tangent spline at the end of it
# directed to a new keypoint.
mapdl.clear(); mapdl.prep7()

k0 = mapdl.k("", 0, 0, 0)
k1 = mapdl.k("", 0, 0, 1)
k2 = mapdl.k("", -1, 1.5, 0)
carc = mapdl.circle(k0, 1, k1, arc=90)
lnum = mapdl.ltan(carc[0], k2)
mapdl.lplot(background='w', color='k', line_width=5, cpos='xy')


###############################################################################
# APDL Command: SPLINE
# ~~~~~~~~~~~~~~~~~~~~
# Generate a segmented spline through 5 keypoints.
mapdl.clear(); mapdl.prep7()

k0 = mapdl.k('', 0, 0, 0)
k1 = mapdl.k('', 0.2, 0.2, 0)
k2 = mapdl.k('', 0.4, 0.3, 0)
k3 = mapdl.k('', 0.6, 0.5, 0)
k4 = mapdl.k('', 0.8, 0.3, 0)
mapdl.spline(k0, k1, k2, k3, k4)
mapdl.lplot(background='w', color='k', line_width=5, cpos='xy')


###############################################################################
# APDL Command: BSPLIN
# ~~~~~~~~~~~~~~~~~~~~
# Generate a single line from a spline fit to a series of keypoints.
# Generate through ``(0, 0, 0)``, ``(0, 1, 0)`` and ``(1, 2, 0)``
#
# This is different than the ``spline`` method as this creates a
# single line rather than multiple lines.
mapdl.clear(); mapdl.prep7()

k0 = mapdl.k("", 0, 0, 0)
k1 = mapdl.k("", 0, 1, 0)
k2 = mapdl.k("", 1, 2, 0)
lnum = mapdl.bsplin(k0, k1, k2)
mapdl.lplot(background='w', color='k', line_width=5, cpos='xy')


###############################################################################
# Line IDs
# ~~~~~~~~
# Return an array of the line IDs
lnum = mapdl.geometry.lnum
lnum


###############################################################################
# Line Geometry
# ~~~~~~~~~~~~~
# Get the VTK ``PolyData`` containing lines.  This VTK mesh can be
# saved or plotted.  For more details, visit https://docs.pyvista.com
lines = mapdl.geometry.lines
lines


###############################################################################
# Line Selection
# ~~~~~~~~~~~~~~
# There are two approaches for selecting lines, the old "legacy"
# style and the new style.  The old style is valuable for those who
# are comfortable with the existing MAPDL commands, and new style is
# useful for selecting lines in a pythonic manner.
#
# This example generates a series of random lines and selects them
mapdl.clear(); mapdl.prep7()


def generate_random_line():
    k0 = mapdl.k('', *np.random.random(3))
    k1 = mapdl.k('', *np.random.random(3))
    mapdl.l(k0, k1)

# create 20 random lines
for _ in range(20):
    generate_random_line()

# Print the line numbers
print(mapdl.geometry.lnum)


###############################################################################
# Select every other line with the old style command.
mapdl.ksel('S', 'KP', '', 1, 20, 2)
print(mapdl.geometry.lnum)


###############################################################################
# Select every other line with the new style command.
#
# Note that the item IDs are 1 based in MAPDL, while Python ranges are
# 0 based.
mapdl.geometry.line_select(range(1, 21, 2))
print(mapdl.geometry.lnum)


###############################################################################
# Select lines from a list
#
# Note that you can ``return_selected`` if you want to see what you
# have selected.  This is helpful when reselecting from existing
# areas.
#
# Note that you could use a numpy array here as well.
items = mapdl.geometry.line_select([1, 5, 10, 20], return_selected=True)
print(items)


###############################################################################
# APDL Command: LPLOT
# ~~~~~~~~~~~~~~~~~~~
# Plot colored lines while displaying the keypoint numbers.
#
# There are a variety of plotting options available for all the common
# plotting methods.

mapdl.lplot(show_keypoint_numbering=True,
            color_lines=True,
            show_line_numbering=False,
            background='black',
            show_bounds=True,
            line_width=5,
            cpos='xy',
            font_size=26)
