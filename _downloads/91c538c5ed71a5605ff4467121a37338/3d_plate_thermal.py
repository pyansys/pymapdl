"""
.. _ref_3d_plate_thermal:

Basic Thermal Analysis with pyMAPDL
-----------------------------------

This example demonstrates how you can use MAPDL to create a plate,
impose thermal boundary conditions, solve, and plot it all within
pyMAPDL.

First, start MAPDL as a service and disable all but error messages.
"""

# sphinx_gallery_thumbnail_number = 2

from ansys.mapdl.core import launch_mapdl
mapdl = launch_mapdl()

###############################################################################
# Geometry and Material Properties
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create a simple beam, specify the material properties, and mesh it.
mapdl.prep7()
mapdl.mp('kxx', 1, 45)
mapdl.et(1, 90)
mapdl.block(-0.3, 0.3, -0.46, 1.34, -0.2, -0.2 + 0.02)
mapdl.vsweep(1)
mapdl.eplot()


###############################################################################
# Boundary Conditions
# ~~~~~~~~~~~~~~~~~~~
# Set the thermal boundary conditions
mapdl.asel('S', vmin=3)
mapdl.nsla()
mapdl.d('all', 'temp', 5)
mapdl.asel('S', vmin=4)
mapdl.nsla()
mapdl.d('all', 'temp', 100)
out = mapdl.allsel()


###############################################################################
# Solve
# ~~~~~
# Solve the thermal static analysis and print the results
mapdl.vsweep(1)
mapdl.run('/SOLU')
print(mapdl.solve())
out = mapdl.finish()


###############################################################################
# Post-Processing using MAPDL
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# View the thermal solution of the beam by getting the results
# directly through MAPDL.
mapdl.post1()
mapdl.set(1, 1)
mapdl.post_processing.plot_nodal_temperature()


###############################################################################
# Alternatively you could also use the result object that reads in the
# result file using pyansys

result = mapdl.result
nnum, temp = result.nodal_temperature(0)
# this is the same as pyansys.read_binary(mapdl._result_file)
print(nnum, temp)
