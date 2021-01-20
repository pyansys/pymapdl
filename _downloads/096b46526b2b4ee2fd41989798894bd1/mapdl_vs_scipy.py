"""
.. _ref_mapdl_math_mapdl_vs_scipy:

Compute Eigenvalues using MAPDL or SciPy
----------------------------------------

This example shows:

- How to extract the stiffness and mass matrices from a MAPDL model.
- How to use the ``Math`` module of PyMapdl to compute the first
  eigenvalues.
- How to can get these matrices in the SciPy world, to get the same
  solutions using Python resources.
- How MAPDL is really faster than SciPy :)
"""

###############################################################################
# First load python packages we need for this example
import time
import math

import matplotlib.pylab as plt
import numpy as np
import scipy
from scipy.sparse.linalg import eigsh

from ansys.mapdl.core import launch_mapdl
from ansys.mapdl.core import examples

###############################################################################
# Next:
#
# - Load the ansys.mapdl module
# - Get the ``Math`` module of PyMapdl
#
mapdl = launch_mapdl()
print(mapdl)
mm = mapdl.math

###############################################################################
# APDLMath EigenSolve
# First load the input file using MAPDL.
#
print(mapdl.input(examples.examples.wing_model))


###############################################################################
# Plot and mesh using the ``eplot`` method.
mapdl.eplot()


###############################################################################
# Next, setup a modal Analysis and request the :math:`K` and math:`M`
# matrices to be formed. MAPDL stores these matrices in a ``.FULL``
# file.

print(mapdl.slashsolu())
print(mapdl.antype(antype='MODAL'))
print(mapdl.modopt(method='LANB', nmode='10', freqb='1.'))
print(mapdl.wrfull(ldstep='1'))

# store the output of the solve command
output = mapdl.solve()


###############################################################################
# Read the sparse matrices using PyMapdl.
#
mapdl.finish()
mm.free()
k = mm.stiff(fname='file.full')
M = mm.mass(fname='file.full')


###############################################################################
# Solve the eigenproblem using PyMapdl with APDLMath.
#
nev = 10
A = mm.mat(k.nrow, nev)

t1 = time.time()
ev = mm.eigs(nev, k, M, phi=A, fmin=1.)
t2 = time.time()
mapdl_elapsed_time = t2-t1
print('\nElapsed time to solve this problem : ', mapdl_elapsed_time)

###############################################################################
# Print eigenfrequencies and accuracy.
#
# Accuracy : :math:`\frac{||(K-\lambda.M).\phi||_2}{||K.\phi||_2}`
#
mapdl_acc = np.empty(nev)

for i in range(nev):
    f = ev[i]                            # Eigenfrequency (Hz)
    omega = 2*np.pi*f                    # omega = 2.pi.Frequency
    lam = omega**2                       # lambda = omega^2

    phi = A[i]                           # i-th eigenshape
    kphi = k.dot(phi)                    # K.Phi
    mphi = M.dot(phi)                    # M.Phi

    kphi_nrm = kphi.norm()               # Normalization scalar value

    mphi *= lam                          # (K-\lambda.M).Phi
    kphi -= mphi

    mapdl_acc[i] = kphi.norm()/kphi_nrm  # compute the residual
    print(f"[{i}] : Freq = {f:8.2f} Hz\t Residual = {mapdl_acc[i]:.5}")


###############################################################################
# Use SciPy to Solve the same Eigenproblem
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# First get MAPDL sparse matrices into the Python memory as SciPy
# matrices.
#
pk = k.asarray()
pm = M.asarray()

# get_ipython().run_line_magic('matplotlib', 'inline')

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('K and M Matrix profiles')
ax1.spy(pk, markersize=0.01)
ax1.set_title('K Matrix')
ax2.spy(pm, markersize=0.01)
ax2.set_title('M Matrix')
plt.show(block=True)


###############################################################################
# Make the sparse matrices for SciPy unsymmetric as symmetric matrices in SciPy
# are memory inefficient.
#
# :math:`K = K + K^T - diag(K)`
#
pkd = scipy.sparse.diags(pk.diagonal())
pK = pk + pk.transpose() - pkd
pmd = scipy.sparse.diags(pm.diagonal())
pm = pm + pm.transpose() - pmd


###############################################################################
# Plot Matrices
#
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('K and M Matrix profiles')
ax1.spy(pk, markersize=0.01)
ax1.set_title('K Matrix')
ax2.spy(pm, markersize=0.01)
ax2.set_title('M Matrix')
plt.show(block=True)


###############################################################################
# Solve the eigenproblem
#
t3 = time.time()
vals, vecs = eigsh(A=pK, M=pm, k=10, sigma=1, which='LA')
t4 = time.time()
scipy_elapsed_time = t4 - t3
print('\nElapsed time to solve this problem : ', scipy_elapsed_time)


###############################################################################
# Convert Lambda values to Frequency values:
# :math:`freq = \frac{\sqrt(\lambda)}{2.\pi}`
#
freqs = np.sqrt(vals) / (2*math.pi)


###############################################################################
# Compute the residual error for SciPy.
#
# :math:`Err=\frac{||(K-\lambda.M).\phi||_2}{||K.\phi||_2}`
#
scipy_acc = np.zeros(nev)

for i in range(nev):
    lam = vals[i]                       # i-th eigenvalue
    phi = vecs.T[i]                     # i-th eigenshape

    kphi = pk*phi.T                     # K.Phi
    mphi = pm*phi.T                     # M.Phi

    kphi_nrm = np.linalg.norm(kphi, 2)  # Normalization scalar value

    mphi *= lam                         # (K-\lambda.M).Phi
    kphi -= mphi

    scipy_acc[i] = 1 - np.linalg.norm(kphi, 2)/kphi_nrm  # compute the residual
    print(f"[{i}] : Freq = {freqs[i]:8.2f} Hz\t Residual = {scipy_acc[i]:.5}")


###############################################################################
# MAPDL is more accurate than SciPy.
#
fig = plt.figure(figsize=(12, 10))
ax = plt.axes()
x = np.linspace(1, 10, 10)
plt.title('Residual Error')
plt.yscale('log')
plt.xlabel('Mode')
plt.ylabel('% Error')
ax.bar(x, scipy_acc, label='SciPy Results')
ax.bar(x, mapdl_acc, label='MAPDL Results')
plt.legend(loc='lower right')
plt.show()

###############################################################################
# MAPDL is faster than SciPy.
#
ratio = scipy_elapsed_time/mapdl_elapsed_time
print(f"Mapdl is {ratio:.3} times faster")
