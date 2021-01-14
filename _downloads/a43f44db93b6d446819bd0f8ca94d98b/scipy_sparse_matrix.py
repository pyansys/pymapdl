"""
Transfer APDLMath Sparse Matrices to and from SciPy Sparse Matrices
-------------------------------------------------------------------

This tutorial will show how to get APDLMath sparse matrices (typically
stiffness or mass matrices extracted from APDL .Full Files) into SciPy
Sparse Matrices.

"""
import matplotlib.pylab as plt

from ansys.mapdl.core.examples import vmfiles
from ansys.mapdl.core import launch_mapdl
mapdl = launch_mapdl()
mm = mapdl.math


################################################################################
# Load and solve verification manual example 153.  Then load the
# stiffness matrix into APDLmath.
out = mapdl.input(vmfiles['vm153'])
k = mm.stiff(fname="PRSMEMB.full")
k

################################################################################
# Copy this APDLMath Sparse Matrix to a SciPy CSR matrix and plot the
# graph of the sparse matrix
pk = k.asarray()
plt.spy(pk)


################################################################################
# You can access the 3 vectors that describe this sparse matrix with.
#
# - ``pk.data``
# - ``pk.indices``
# - ``pk.indptr``
#
# See the ``scipy`` documentation of the csr matrix at `scipy.sparse.csr_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_ for additional details.

print(pk.data[:10])
print(pk.indices[:10])
print(pk.indptr[:10])


################################################################################
# ### Create a APDLMath Sparse Matrix from a SciPy Sparse CSR Matrix
#
# Here, we transfer the ``scipy`` CSR matrix back to MAPDL.  While
# this example uses a matrix that was originally within MAPDL, you can
# load any CSR matrix to MAPDL.

my_mat = mm.matrix(pk, 'my_mat', triu=True)
my_mat

################################################################################
# Check initial matrix ``k`` and ``my_mat`` are exactly the sames:
# We compute the norm of the difference, should be zero

msub = k - my_mat
mm.norm(msub)


################################################################################
# CSR Representation in MAPDL
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Printing the list of objects in the MAPDL space, we find:
#
# - 2 SMAT objects, corresponding to the ``k``, ``MSub`` matrices,
# - with encrypted names
# - The ``my_mat`` SMAT object. Its size is zero, because the 3
# - vectors are stored separately
# - the 3 vectors of the CSR my_mat structure: ``MY_MAT_PTR``, ``MY_MAT_IND``
# - and ``MY_MAT_DATA``

mm.status()


################################################################################
# MAPDL Python Matrix Correspondence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To determine which MAPDL object corresponds to which Python object,
# access the id property of the Python object.

print("name(k)=" + k.id)
print("name(my_mat)=" + my_mat.id)
print("name(msub)=" + msub.id)
