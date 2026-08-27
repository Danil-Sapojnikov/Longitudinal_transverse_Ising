"""
***********************************************************************
* Exact Diagonalisation of the transverse field Ising model for low N. 

* Comparison to known analytic result.

* Inclusion of a longitudinal field.
***********************************************************************
"""

import numpy as np
import matplotlib as plt
import scipy as sp

#----------------------------------------------------------------------

N = 10 # Number of elements in chain (N>=2)
J = 1 # Exchange coupling constant
H_X = 0 # Transverse field
H_Z = 0 # Longitudinal field

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

E,V = sp.sparse.linalg.eigs(Hamiltonian, k = 12)

print(E)