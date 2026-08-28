"""
***********************************************************************
* Exact Diagonalisation of the transverse field Ising model for low N. 

* Comparison to known analytic result.

* Inclusion of a longitudinal field.
***********************************************************************
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

#----------------------------------------------------------------------

N = 4 # Number of elements in chain (N>=2)
J = 1 # Exchange coupling constant
H_X = 0 # Transverse field
H_Z = 0 # Longitudinal field

NUM_EVALS = 4
TOL = 10e-10

SAVETEXT = False
SAVEFIG = False

# The identity matrix and 4 Pauli matrices in sparse form
I = sp.sparse.csr_array(np.array([[1,0],[0,1]]))
X = sp.sparse.csr_array(np.array([[0,1],[1,0]]))
Y = sp.sparse.csr_array(np.array([[0,-1j],[1j,0]]))
Z = sp.sparse.csr_array(np.array([[1,0],[0,-1]]))

#----------------------------------------------------------------------

sigmaZ_list = []

for i in range(0,N):
    if i == 0:
        sigmaZ_i = Z
    else:
        sigmaZ_i = I

    for k in range(1,N):
        if k == i:
            sigmaZ_i = sp.sparse.kron(sigmaZ_i,Z)
        else:
            sigmaZ_i = sp.sparse.kron(sigmaZ_i,I)

    sigmaZ_list.append(sigmaZ_i) #add .toarray() for nicer layout

#print(sigmaZ_list)

sigmaX_list = []

for i in range(0,N):
    if i == 0:
        sigmaX_i = X
    else:
        sigmaX_i = I

    for k in range(1,N):
        if k == i:
            sigmaX_i = sp.sparse.kron(sigmaX_i,X)
        else:
            sigmaX_i = sp.sparse.kron(sigmaX_i,I)

    sigmaX_list.append(sigmaX_i)

#----------------------------------------------------------------------

exchange = sp.sparse.csr_array(np.zeros((2**N,2**N)))
for i in range(0,N-1):
    exchange = exchange + sigmaZ_list[i]@sigmaZ_list[i+1]
exchange = J * (exchange + sigmaZ_list[N-1]@sigmaZ_list[0])

transverse = sp.sparse.csr_array(np.zeros((2**N,2**N)))
for i in range(0,N):
    transverse = transverse + sigmaX_list[i]
transverse *= H_X

longitudinal = sp.sparse.csr_array(np.zeros((2**N,2**N)))
for i in range(0,N):
    longitudinal = longitudinal + sigmaX_list[i]
longitudinal *= H_Z

Hamiltonian = -exchange - transverse - longitudinal

#print(Hamiltonian.toarray())

#----------------------------------------------------------------------

evals,evecs = sp.sparse.linalg.eigs(Hamiltonian, k = NUM_EVALS, which = 'SR')

#print(evals_rounded)
#print(np.abs(evecs[:, 3])**2)
#print(np.sum(np.abs(evecs[:, 3])**2))

#print(evals)
#print(len(evals))
#print(np.real(evecs_rounded[:,0]))

#----------------------------------------------------------------------

from itertools import product

spin_configs = list(product([1, -1], repeat=N))

for n in range(NUM_EVALS):
    print(f"\nEigenvalue {n}: E = {evals[n].real:.6f}")

    for i, config in enumerate(spin_configs):
        coefficient = evecs[i, n]
        probability = abs(coefficient)**2

        if probability > 1e-10:
            print(
                config,
                #f"coefficient = {coefficient:.6f}",
                f"probability = {probability:.6f}"
            )

#----------------------------------------------------------------------
