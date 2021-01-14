"""
.. _ref_mapdl_math_basic:

PyMAPDL APDLMath Basic Operations
---------------------------------

This tutorial shows how you can use pymapdl to use APDL math for basic
operations on APDLMath vectors and matrices in the APDL memory
workspace.

The `ansys.mapdl.math` submodule gives access to APDLMath features
inside PyMAPDL.

"""
import numpy as np

from ansys.mapdl.core import launch_mapdl

# Start MAPDL as a service and create an APDLMath object.
mapdl = launch_mapdl()
mm = mapdl.math


###############################################################################
# Create and Manipulate Vectors
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create 2 APDLMath vectors of size 5. :math:`\vec{v}` is initialized with
# ones, $\vec{w}$ is filled with random values
#
# Corresponding APDLMath commands
# - `*VEC,V,D,ALLOC,5`
# - `*INIT,V,CONST,1`
# - `*VEC,W,D,ALLOC,5`
# - `*INIT,W,RAND`

v = mm.ones(5)
w = mm.rand(5)
print(w)


###############################################################################
# Use operators on vectors
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Just like `numpy` PyMAPDL APDLMath vectors can be have most of the
# standard operators (e.g. ``+, -, +=, -=, *=``)
#
# Here we form :math:`\vec{z}=\vec{v}+\vec{w}`
#
# Then we compute :math:`\|z\|_2` (the default `norm` is nrm2, but you
# can use `.norm('nrm1')` or `.norm('nrminf')` for different normals.
# See `help(z.norm)` for additional details.
#
# APDLMath Commands:
# - `*VEC,Z,D,COPY,V`
# - `*AXPY,1,,W,1,,Z`
# - `*NRM,Z,,nrmval`

z = v + w
z.norm()


###############################################################################
# Methods
# ~~~~~~~
# Alternatively you can use methods, following the numpy
# standards. Available methods are:
#
# - `mm.add()`
# - `mm.subtract()`
# - `mm.dot()`
#
# Equivalent operator:
# `z = v + w`
#
# Equivalent APDLMath Commands:
# - `*VEC,Z,D,COPY,V`
# - `*AXPY,1,,W,1,,Z`
z = mm.add(v, w)
z.norm()

###############################################################################
# Subtraction
#
# Equivalent operator:
# z = v - w
#
# Equivalent APDLMath Commands:
# - `*VEC,Z,D,COPY,V`
# - `*AXPY,-1,,W,1,,Z`
z = mm.subtract(v, w)
print(z)


###############################################################################
# Dot product of 2 vectors
#
# Equivalent APDLMath Command: `*DOT,V,W,dotval`

vw = mm.dot(v, w)
print("Dot product :", str(vw))


###############################################################################
# Perform an in-place operations (without copying vectors)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In-Place Addition
#
# MAPDL Commands:
# - `*AXPY,1,,V,1,,Z`
# - `*PRINT,Z`
v += v
print(v)


###############################################################################
# In-Place Multiplication
#
# MAPDL Command: `*SCAL,v,2`
v *= 2
print(v)

###############################################################################
# In-Place Multiplication
#
v /= 2.
print(v)


###############################################################################
# Working with Dense Matrices
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Allocate two dense matrices with random values.
#
# MAPDL Commands:
#
# - `*DMAT,m1,D,ALLOC,4,5`
# - `*INIT,m1,RAND`
# - `*DMAT,m1,D,ALLOC,4,5`
# - `*INIT,m1,CONST,1`

m1 = mm.rand(4, 5)
m2 = mm.ones(4, 5)
m1, m2

###############################################################################
# **Add** these 2 dense matrices, and **scale** the result matrix.
#
# Mapdl Commands
# - `*DMAT,m3,D,COPY,m1`
# - `*AXPY,1,,m2,1,,m3`
m3 = m1 + m2
print(m3)

m3 *= 2
print(m3)

###############################################################################
# ***Transpose*** a Matrix
#
m4 = m3.T
print(m4)


###############################################################################
# As for vectors, methods are also available as an alternative to operators.
m3 = mm.add(m1, m2)
print(m3)


###############################################################################
# Compute a matrix vector multiplication
#
mw = m3.dot(m4)
print(mw)


###############################################################################
# APDLMath matrices can be identified by printing, viewing their types, or with using the `__repr__` method by simply typing out the variable
#
# APDLMath Matrix
# ~~~~~~~~~~~~~~~
type(m1)
print(m1)
m1


###############################################################################
# APDLMath Vector
#
type(w)
print(w)
w

###############################################################################
# Numpy methods on APDLMath objects
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Regardless of the underlying APDLMath object type, you are generally
# able to perform most numpy or scipy operations on these arrays.  You
# can do this one of two ways.  First, you can convert a matrix to a numpy array:
apdl_mat = mm.rand(5, 5)
np_mat = apdl_mat.asarray()
print(np_mat)


###############################################################################
# Alternatively, you can simply use numpy to compute the max of the array
#
# This works because PyMAPDL copies over the matrix to the local
# python memory and then computes the max using numpy.
print(np.max(apdl_mat))


###############################################################################
# This works for most numpy operations, but keep in mind that
# operations that are supported within MAPDL (such as adding or
# multiplying arrays) will compute much faster as the data is not copied.
#
apdl_arr = mm.rand(5, 5)
np_array = apdl_mat.asarray()
print(np.allclose(apdl_mat, np_array))
