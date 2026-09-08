"""
***********************************************************************

*Tutorial QuSpin basics tutorial

***********************************************************************
"""

#import sys, os
# set options and import required packages
#os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
#os.environ['OMP_NUM_THREADS']='4' # set number of OpenMP threads to run in parallel
#os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel

import quspin
import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(suppress=True, precision=3) # suppress machine precision; print numbers up to three decimals

#---------------------------------------------------------------------------------------------------------------

from quspin.basis import spin_basis_1d

basis_1 = spin_basis_1d(L=1) # single spin-1/2 particle / qubit / two-level system
#print(basis_1)

# define state |0> = |down>
psi_down = np.array([0.0,1.0])
#print('\n|down> = {0}'.format(psi_down) )

# define state |1> = |up>
psi_up = np.array([1.0,0.0])
#print('\n|up> = {0}'.format(psi_up) )

# define a superposition state: (|0>+|1>)/sqrt(2)
psi_plus = np.array([1.0,1.0])/np.sqrt(2)
#print(f'\n|+> = 1/sqrt(2)(|0>+|1>) = {psi_plus}')

# compute overlap <up|+>
overlap = psi_up.conj() @ psi_plus
#print(f'\n<up|+> = {overlap}')

# compute overlap squared |<up|+>|^2
#print(f'\n|<up|+>|^2 = {np.abs(overlap)**2}')

#---------------------------------------------------------------------------------------------------------------

from quspin.operators import hamiltonian

# define coupling strengths
hx, hy, hz = 1.0, 1.0, 1.0

# associate each coupling to the spin labeled 0
hx_list = [[hx,0],] # coupling: hx multiplies spin labeled 0
hy_list = [[hy,0],] # coupling: hy multiplies spin labeled 0
hz_list = [[hz,0],] # coupling: hz multiplies spin labeled 0

# associate a Pauli matrix to each coupling list
static_terms = [['x',hx_list], # assign coupling list hx_list to Pauli operator sigma^x
    		    ['y',hy_list], # assign coupling list hy_list to Pauli operator sigma^y
    		    ['z',hz_list], # assign coupling list hz_list to Pauli operator sigma^z
    		   ]
dynamic_terms = [] # ignore this for the time being (we'll come to it later)

# construct Hamiltonian in the single-spin basis defined above
H_1 = hamiltonian(static_terms, dynamic_terms, basis=basis_1,)

#print(H_1)
#print('cast as a dense array:\n{}\n'.format(H_1.toarray()))

# compute expectation value <down|H|down>
expt_H_1_psi = H_1.expt_value(psi_down)

#print('\n<down|H|down> = {}'.format(expt_H_1_psi) )

#---------------------------------------------------------------------------------------------------------------

### two coupled spins / qubits
basis_2 = spin_basis_1d(L=2)
print(basis_2)

