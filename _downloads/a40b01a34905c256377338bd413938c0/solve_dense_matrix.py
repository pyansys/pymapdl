"""
Use APDLMath to Solve a Dense Matrix Linear System
--------------------------------------------------
Use the APDLMath module to solve a Dense Matrix Linear System

"""

import time
import numpy.linalg as np
from ansys.mapdl.core import launch_mapdl

# Start MAPDL as a service and disable all but error messages.
# Create an APDLMath object.
mapdl = launch_mapdl()
mm = mapdl.math


###############################################################################
# Allocate a Dense Matrix in the APDLMath workspace
#
mapdl.clear()
dim = 10000
a = mm.rand(dim, dim)
b = mm.rand(dim)
x = mm.zeros(dim)

###############################################################################
# We copy matrices into numpy arrays before they are modified by factorization call
#
a_py = a.asarray()
b_py = b.asarray()

###############################################################################
# Solve using APDLMath
#
print(f'Solving a ({dim} x {dim}) dense linear system using MAPDL...')

t1 = time.time()
s = mm.factorize(a)
x = s.solve(b, x)
t2 = time.time()
print(f'Elapsed time to solve the linear system using Mapdl: {t2 - t1} seconds')

###############################################################################
# Norm of the MAPDL Solution
mm.norm(x)


###############################################################################
# Solve the solution using numpy
#
print(f'Solving a ({dim} x {dim}) dense linear system using numpy...')

t1 = time.time()
x_py = np.linalg.solve(a_py, b_py)
t2 = time.time()
print(f'Elapsed time to solve the linear system using numpy: {t2 - t1} seconds')

###############################################################################
# Norm of the numpy Solution
#
np.linalg.norm(x_py)
