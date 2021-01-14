"""
.. _ref_thermal_transient:

Example Thermal Transient Analysis
----------------------------------
This example shows how you can use PyMAPDL to input a time dependent
temperature table to vary the temperature at a beam.  This uses
convection loads with independently varying convection coefficient and
bulk temperature.

Example adapted from:
https://www.simutechgroup.com/tips-and-tricks/fea-articles/97-fea-tips-tricks-thermal-transient

Thanks SimuTech!

"""
# sphinx_gallery_thumbnail_number = 4

import numpy as np

import matplotlib.pyplot as plt
from ansys.mapdl.core import launch_mapdl

mapdl = launch_mapdl(loglevel='ERROR')

mapdl.clear()
mapdl.prep7()

# Material properties-- 1020 steel in imperial
mapdl.units('BIN')  # U.S. Customary system using inches (in, lbf*s2/in, s, °F).
mapdl.mp('EX', 1, 30023280.0)
mapdl.mp('NUXY', 1, 0.290000000)
mapdl.mp('ALPX', 1, 8.388888889E-06)
mapdl.mp('DENS', 1, 7.346344000E-04)
mapdl.mp('KXX', 1, 6.252196000E-04)
mapdl.mp('C', 1, 38.6334760)

# use a thermal element type
mapdl.et(1, 'SOLID70')


###############################################################################
# Geometry and Mesh
# ~~~~~~~~~~~~~~~~~
# Create a block 5x1x1 inches in size and mesh it
mapdl.block(0, 5, 0, 1, 0, 1)
mapdl.lesize('ALL', 0.2, layer1=1)

mapdl.mshape(0, '3D')
mapdl.mshkey(1)
mapdl.vmesh(1)
mapdl.eplot()


###############################################################################
# Setup the Solution
# ~~~~~~~~~~~~~~~~~~
# Solve a transient analysis while ramping the load up and down.
#
# Note the solution time commands in the above code fragment. The
# final TIME is set to 1000 seconds. Time substep size is permitted to
# range from a minimum of 2 seconds to a maximum of 50 seconds in the
# DELTIM command. A first substep of 10 seconds is applied. Automatic
# time substep sizing will vary substeps between the extremes.
#
# A Table Array is used for the time-dependent Convection Coefficient
# values. Times go in the Zeroth column, while associated Convection
# Coefficients go in the First column.

mapdl.run('/SOLU')
mapdl.antype(4)            # transient analysis
mapdl.trnopt('FULL')       # full transient analysis
mapdl.kbc(0)               # ramp loads up and down

# Time stepping
end_time = 1500
mapdl.time(end_time)       # end time for load step
mapdl.autots('ON')         # use automatic time stepping


# setup where the subset time is 10 seconds, time
mapdl.deltim(10, 2, 25)    # substep size (seconds)
#                          -- minimum value shorter than smallest
#                            time change in the table arrays below

# Create a table of convection times and coefficients and transfer it to MAPDL
my_conv = np.array([[0, 0.001],      # start time
                    [120, 0.001],    # end of first "flat" zone
                    [130, 0.005],    # ramps up in 10 seconds
                    [700, 0.005],    # end of second "flat zone
                    [710, 0.002],    # ramps down in 10 seconds
                    [end_time, 0.002]])  # end of third "flat" zone
mapdl.load_table('my_conv', my_conv, 'TIME')


# Create a table of bulk temperatures for a given time and transfer to MAPDL
my_bulk = np.array([[0, 100],      # start time
                    [120, 100],    # end of first "flat" zone
                    [500, 300],    # ramps up in 380 seconds
                    [700, 300],    # hold temperature for 200 seconds
                    [900, 75],     # temperature ramps down for 200 seconds
                    [end_time, 75]])   # end of second "flat" zone
mapdl.load_table('my_bulk', my_bulk, 'TIME')


###############################################################################
# The Transient Thermal Solve
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This model is to be solved in one time step. For this reason, a
# ``TSRES`` command is used for force the solver to include a
# ``SOLVE`` at every time point in the two Table Arrays above. This
# ensures that the time-dependent curves are followed by the transient
# analysis. Intermediate solutions between the ``TSRES`` time points
# will be included according to the ``DELTIM`` command and the
# automatic time stepping decisions of the ANSYS solver.
#
# In this example, the times for the ``TSRES`` array illustrated above
# have been determined manually. A set of APDL commands could be used
# to automate this process for chosen Table Array entries, in more
# complex modeling situations, including checks that no time intervals
# are too short.
#
# Results at substeps will be wanted if the intermediate solutions of
# the time-transient analysis are to be available for post-processing
# review. The ``OUTRES`` command is used to control how much is written to
# the results file. In this example the OUTRES command will be used to
# simply write out all results for all substeps. In work with large
# models and may substeps, too much data will be written if such a
# strategy is employed for ``OUTRES``, and other options will need to be
# considered. Note that one option for the ``OUTRES`` command is to
# control times at which results are written with a Table Array, much
# as is used in the ``TSRES`` command, but typically for a larger number
# of time points, although including those of the TSRES array.
#
# The initial condition starting temperature is controlled for this
# example with the ``TUNIF`` command. Note that thermal transient
# analyses can also have a starting temperature profile formed by a
# static thermal ``SOLVE``. If a user neglects to set an initial
# temperature in ANSYS Mechanical APDL, a value of zero will be used,
# which is often not what is desired.
#
# The thermal convective loads are applied with an SF family
# command—in this example a convective load is applied to the end face
# of the solid model by the SFA command, using the Table Array entries
# for convection and bulk temperature that were developed above. The
# Table Array names are surrounded with percent signs (%).  A SOLVE is
# then performed.

# Force transient solve to include the times within the conv and bulk arrays
# my_tres = np.unique(np.vstack((my_bulk[:, 0], my_conv[:, 0])))[0]  # same as
mapdl.parameters['my_tsres'] = [120, 130, 500, 700, 710, 900, end_time]
mapdl.tsres('%my_tsres%')

mapdl.outres('ERASE')
mapdl.outres('ALL', 'ALL')

mapdl.eqslv('SPARSE')  # use sparse solver
mapdl.tunif(75)        # force uniform starting temperature (otherwise zero)

# apply the convective load (convection coefficient plus bulk temperature)
# use "%" around table array names
mapdl.sfa(6, 1, 'CONV', '%my_conv%', ' %my_bulk%')

# solve
mapdl.solve()

###############################################################################
# Post-Processing
# ~~~~~~~~~~~~~~~
# Open up the result file using ``ansys.mapdl.reader``
# result = mapdl.thermal_result
mapdl.post1()

###############################################################################
# Visualize a Slice
# ~~~~~~~~~~~~~~~~~
# Visualize a slice through the dataset using ``pyvista``
# for more details visit <https://docs.pyvista.org/>`_.

# get the temperature of the 30th result set
mapdl.set(1, 30)
temp = mapdl.post_processing.nodal_temperature

# Load this result into the underlying VTK grid
grid = mapdl.mesh._grid
grid['temperature'] = temp

# generate a single horizontal slice slice along the XY plane
single_slice = grid.slice(normal=[0, 0, 1], origin=[0, 0, 0.5])
single_slice.plot(scalars='temperature')


###############################################################################
# Visualize Several Slices
# ~~~~~~~~~~~~~~~~~~~~~~~~
# This shows how you can visualize a series of slices through a dataset

# get the temperature of a different result set
mapdl.set(1, 120)
temp = mapdl.post_processing.nodal_temperature

# Load this result into the underlying VTK grid
grid = mapdl.mesh._grid
grid['temperature'] = temp

# generate a single horizontal slice slice along the XY plane
slices = grid.slice_along_axis(7, 'y')
slices.plot(scalars='temperature', lighting=False, show_edges=True)


###############################################################################
# Temperature at a Single Point
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Extract the temperature at a single node and plot it with respect to
# the input temperatures using ``ansys.mapdl``
#
# Here, we use the ``get_value`` command which is very similar to the
# ``*GET`` command, except it immediately returns the value as a
# python accessible variable, rather than storing it to a MAPDL value.

# for example, the temperature of Node 12 is can be retrieved simply with:
mapdl.get_value('node', 12, 'TEMP')

# note that this is similar to # *GET, Par, NODE, N, Item1, IT1NUM, Item2, IT2NUM
# See the MAPDL reference for all the items you can obtain using *GET

###############################################################################
# Here, we extract the temperature of the node across for each solution
nsets = mapdl.post_processing.nsets
node_temp = []
for i in range(1, 1 + nsets):
    mapdl.set(1, i)
    node_temp.append(mapdl.get_value('node', 12, 'TEMP'))

# here are the first 10 temperatures
node_temp[:10]

###############################################################################
# Alternatively, you can simply grab the data for the node from the
# entire response.  This is less efficient as the entire data set is
# sent back for each result.

# get the index of node 12 in MAPDL
idx = np.nonzero(mapdl.mesh.nnum == 12)[0][0]

# get the temperature at that index for each result
node_temp_from_post = []
for i in range(1, 1 + nsets):
    mapdl.set(1, i)
    node_temp_from_post.append(mapdl.post_processing.nodal_temperature[idx])

# Again, the first 10 temperatures
node_temp_from_post[:10]

###############################################################################
# Plot the temperature as a function of time
time_values = mapdl.post_processing.time_values
plt.plot(time_values, node_temp, label='Node 12')
plt.plot(my_bulk[:, 0], my_bulk[:, 1], ':', label='Input')
plt.legend()
plt.xlabel('Time (seconds)')
plt.ylabel('Temperature ($^\circ$F)')
plt.show()
