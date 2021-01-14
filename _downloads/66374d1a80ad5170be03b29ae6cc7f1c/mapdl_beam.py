"""
.. _ref_mapdl_beam:

MAPDL 2D Beam Example
---------------------

This is an example from FINITE ELEMENT ANALYSIS USING ANSYS 11.0

"""

###############################################################################
# Launch MAPDL with interactive plotting
from ansys.mapdl.core import launch_mapdl
mapdl = launch_mapdl()

###############################################################################
# Define an I-beam
mapdl.prep7()
mapdl.et(1, 'BEAM188')
mapdl.keyopt(1, 4, 1)  # transverse shear stress output

#material properties
mapdl.mp('EX', 1, 2E7)  # N/cm2
mapdl.mp('PRXY', 1, 0.27)  #  Poisson's ratio

# beam properties in centimeters
sec_num = 1
mapdl.sectype(sec_num, 'BEAM', 'I', 'ISection', 3)
mapdl.secoffset('CENT')
beam_info = mapdl.secdata(15, 15, 29, 2, 2, 1)  # dimensions are in centimeters

###############################################################################
# Create nodes within MAPDL
mapdl.n(1, 0, 0, 0)
mapdl.n(12, 110, 0, 0)
mapdl.n(23, 220, 0, 0)
mapdl.fill(1, 12, 10)
mapdl.fill(12, 23, 10)

# list the node coordinates
print(mapdl.mesh.nodes)

# list the node numbers
print(mapdl.mesh.nnum)

# plot the nodes using VTK
mapdl.nplot(vtk=True, knum=True, cpos='xy', show_bounds=True, point_size=10)

###############################################################################
# create elements between the nodes
# we can just manually create elements since we know that the elements
# are sequential
for node in mapdl.mesh.nnum[:-1]:
    mapdl.e(node, node + 1)

# print the elements from MAPDL
print(mapdl.elist())

###############################################################################
# Access them as a list of arrays
# See the documentation on ``mapdl.mesh.elem`` for interperting the
# individual elements
for elem in mapdl.mesh.elem:
    print(elem)



###############################################################################
# Define the boundary conditions

# Allow movement only in the X and Z direction
for const in ['UX', 'UY', 'ROTX', 'ROTZ']:
    mapdl.d('all', const)

# constrain just nodes 1 and 23 in the Z direction
mapdl.d(1, 'UZ')
mapdl.d(23, 'UZ')

# apply a -Z force at node 12
mapdl.f(12, 'FZ', -22840)


###############################################################################
# run the static analysis
mapdl.run('/solu')
mapdl.antype('static')
print(mapdl.solve())
