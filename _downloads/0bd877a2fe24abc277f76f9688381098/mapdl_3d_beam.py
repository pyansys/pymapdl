"""
.. _ref_3d_beam:

MAPDL 3D Beam Example
---------------------

This is a simple example that loads an archive file containing a beam
and then runs a modal analysis using the simplified ``modal_analysis``
method.

First, start by launching MAPDL as a service.
"""

from ansys.mapdl.reader import examples
from ansys.mapdl.core import launch_mapdl
mapdl = launch_mapdl()

# load a beam stored as an example archive file and mesh it
mapdl.cdread('db', examples.hexarchivefile)
mapdl.esel('s', 'ELEM', vmin=5, vmax=20)
mapdl.cm('ELEM_COMP', 'ELEM')
mapdl.nsel('s', 'NODE', vmin=5, vmax=20)
mapdl.cm('NODE_COMP', 'NODE')

# boundary conditions
mapdl.allsel()

# dummy steel properties
mapdl.prep7()
mapdl.mp('EX', 1, 200E9)  # Elastic moduli in Pa (kg/(m*s**2))
mapdl.mp('DENS', 1, 7800)  # Density in kg/m3
mapdl.mp('NUXY', 1, 0.3)  # Poissons Ratio
mapdl.emodif('ALL', 'MAT', 1)

# fix one end of the beam
mapdl.nsel('S', 'LOC', 'Z')
mapdl.d('all', 'all')
mapdl.allsel()

mapdl.mxpand(elcalc='YES')
mapdl.modal_analysis(nmode=6)


###############################################################################
# View the results using the pyansys result object
result = mapdl.result
print(result)


###############################################################################
# Access nodal displacement values
nnum, disp = result.nodal_displacement(0)

# print the nodes 50 - 59
for i in range(49, 59):
    print(nnum[i], disp[i])


###############################################################################
# Plot a modal result
result.plot_nodal_displacement(0, show_edges=True)


###############################################################################
# Animate a modal result
# result.animate_nodal_solution(0, show_edges=True, loop=False, displacement_factor=10,
                              # movie_filename='demo.gif')

