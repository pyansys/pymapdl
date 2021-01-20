"""
.. _ref_mapdl_math_eigen_solve:

Using APDLMath to solve Eigenproblems
-------------------------------------
Use APDLMath to solve eigenproblems.

This example uses a verification manual input file, but you can use
your own sparse or dense matrices and solve those.

"""
import matplotlib.pylab as plt
import time
import numpy as np

from ansys.mapdl.core.examples import vmfiles
from ansys.mapdl.core import launch_mapdl

# Start MAPDL as a service and create an APDLMath object
mapdl = launch_mapdl(loglevel='ERROR')
mm = mapdl.math


###############################################################################
# First we get the `STIFF` and `MASS` matrices from the full file
# after running the input file from Verification Manual 153
#
out = mapdl.input(vmfiles['vm153'])

k = mm.stiff(fname="PRSMEMB.full")
m = mm.mass(fname="PRSMEMB.full")


###############################################################################
# Display size of the M and K matrices
print(m.shape)
print(k.shape)

###############################################################################
# Allocate an array to store the eigenshapes.
# where `nev` is the number of eigenvalues requested
#
nev = 10
a = mm.mat(k.nrow, nev)
a

###############################################################################
# Perform the the modal analysis.
#
# The algorithm is automatically chosen with respect to the matrices
# properties (e.g. scalar, storage, symmetry...)
#
print('Calling MAPDL to solve the eigenproblem...')

t1 = time.time()
ev = mm.eigs(nev, k, m, phi=a)
print(f'Elapsed time to solve this problem: {time.time() - t1}')


###############################################################################
# This is the vector of eigenfrequencies.
print(ev)

###############################################################################
# Verify the accuracy of eigenresults
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check the residual error for the first eigenresult
# :math:`R_1=||(K-\lambda_1.M).\phi_1||_2`
#
# First, we compute :math:`\lambda_1 = \omega_1^2 = (2.\pi.f_1)^2`

# Eigenfrequency (Hz)
i = 0
f = ev[0]
omega = 2*np.pi*f
lam = omega*omega


###############################################################################
# Then we get the 1st Eigenshape :math:`\phi_1`, and compute
# :math:`K.\phi_1` and :math:`M.\phi_1`

# shape
phi = a[0]

# APDL Command: *MULT,K,,Phi,,KPhi
kphi = k.dot(phi)

# APDL Command: *MULT,M,,Phi,,MPhi
mphi = m.dot(phi)


######################################################################
# Next, compute the `math`:||K.\phi_1||_2 quantity and normalize the
# residual value.

# APDL Command: *MULT,K,,Phi,,KPhi
kphi = k.dot(phi)


# APDL Command: *NRM,KPhi,NRM2,KPhiNrm
kphinrm = kphi.norm()


###############################################################################
# Then we add these two vectors, using the :math:`\lambda_1` scalar
# factor and finally compute the normalized residual value
# :math:`\frac{R_1}{||K.\phi_1||_2}`

# APDL Command: *AXPY,-lambda,,MPhi,1,,KPhi
mphi *= lam
kphi -= mphi

# Compute the residual
res = kphi.norm()/kphinrm
print(res)

###############################################################################
# This residual can be computed for all eigenmodes
#

def get_res(i):
    """Compute the residual for a given eigenmode"""
    # Eigenfrequency (Hz)
    f = ev[i]

    # omega = 2.pi.Frequency
    omega = 2*np.pi*f

    # lambda = omega^2
    lam = omega*omega

    # i-th eigenshape
    phi = a[i]

    # K.Phi
    kphi = k.dot(phi)

    # M.Phi
    mphi = m.dot(phi)

    # Normalization scalar value
    kphinrm = kphi.norm()

    # (K-\lambda.M).Phi
    mphi *= lam
    kphi -= mphi

    # return the residual
    return kphi.norm()/kphinrm


mapdl_acc = np.zeros(nev)

for i in range(nev):
    f = ev[i]
    mapdl_acc[i] = get_res(i)
    print(f'[{i}] : Freq = {f}\t - Residual = {mapdl_acc[i]}')

###############################################################################
# Plot Accuracy of Eigenresults

fig = plt.figure(figsize=(12, 10))
ax = plt.axes()
x = np.linspace(1, nev, nev)
plt.title('APDL Math Residual Error (%)')
plt.yscale('log')
plt.ylim([10E-13, 10E-7])
plt.xlabel('Frequency #')
plt.ylabel('Errors (%)')
ax.bar(x, mapdl_acc, label='MAPDL Results')
