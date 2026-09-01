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

N = 8 # Number of elements in chain (N>=2)
J = 1 # Exchange coupling constant
H_X = 0.3 # Transverse field
H_Z = 0.1 # Longitudinal field

NUM_EVALS = 100 # Number of Eigenvalues calculated
#TOL = 1e-10
NUM_BINS = 10

SAVETEXT = False
SAVEFIG = False
FIGNAME = 'Test.png'

# The identity matrix and 4 Pauli matrices in sparse form
I = sp.sparse.csr_array(np.array([[1,0],[0,1]]))
X = sp.sparse.csr_array(np.array([[0,1],[1,0]]))
Y = sp.sparse.csr_array(np.array([[0,-1j],[1j,0]]))
Z = sp.sparse.csr_array(np.array([[1,0],[0,-1]]))

#----------------------------------------------------------------------
# Assemble the Z_i and X_i arrays relevant within the Hamiltonian for the corresponding length of chain

def assembleZ_i():

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

        sigmaZ_list.append(sigmaZ_i) #add .toarray() for nicer print layout

    #print(sigmaZ_list)
    return sigmaZ_list

def assembleX_i():

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

    #print(sigmaX_list)
    return sigmaX_list

#----------------------------------------------------------------------
# Assemble the Longitudinal + Transverse field Ising Hamiltonian

def assmemble_Hamiltonian(sigmaZ_list,sigmaX_list,h_z,h_x):

    exchange = sp.sparse.csr_array(np.zeros((2**N,2**N)))
    for i in range(0,N-1):
        exchange += sigmaZ_list[i]@sigmaZ_list[i+1]
    exchange = J * (exchange + sigmaZ_list[N-1]@sigmaZ_list[0])

    transverse = sp.sparse.csr_array(np.zeros((2**N,2**N)))
    for i in range(0,N):
        transverse += sigmaX_list[i]
    transverse *= h_x

    longitudinal = sp.sparse.csr_array(np.zeros((2**N,2**N)))
    for i in range(0,N):
        longitudinal += sigmaZ_list[i]
    longitudinal *= h_z

    Hamiltonian = -exchange - transverse - longitudinal

    #print(Hamiltonian.toarray())
    return Hamiltonian

#----------------------------------------------------------------------
# Find the Eigenvalues and Eigenvectors of the Hamiltonian, then the energy spacings

def find_eigs(Hamiltonian):

    evals,evecs = sp.sparse.linalg.eigs(Hamiltonian, k = NUM_EVALS, which = 'SR')

    return evals,evecs

def find_spacings(eigvals):

    eigvals = np.sort(eigvals)
    eigvals_shifted = np.insert(eigvals, 0, eigvals[0])
    eigvals_shifted = np.delete(eigvals_shifted, -1)

    eigvals_space = eigvals - eigvals_shifted

    mean = np.mean(eigvals_space)
    eigvals_space_scaled = eigvals_space/mean

    return (eigvals_space,eigvals_space_scaled)

#----------------------------------------------------------------------
# Output functions

def print_eigs(evals,evecs):

    for i in range(len(evals)):
        print(f"\nEigenvalue {i+1}")
        print(f"{evals[i].real:.6f}")
        print(f"{np.round(np.real(evecs[:,i]),3)}")

def poisson_dist(x):

    return np.exp(-x)

def wigner_dist(x):

    power = -np.pi / 4 * x**2

    return np.pi/2 * x * np.exp(power)

def spacings_hist(scaled_spacings):

    x = np.linspace(0,5,1000)
    scaled_spacings = np.delete(scaled_spacings, np.where(scaled_spacings > 10*np.mean(scaled_spacings)))

    plt.hist(scaled_spacings, bins=NUM_BINS, density = True, label = 'Calulated Differences')
    plt.plot(x, wigner_dist(x), label = 'GOE')
    plt.plot(x, poisson_dist(x), label = 'Poisson Distribution', color = 'b')
    
    plt.legend()
    plt.title('Energy Gap Probability Distribution')
    plt.xlabel(r'$\frac{s}{<s>}$')
    plt.ylabel('Probability Density')
    
    if SAVEFIG:
        plt.savefig(FIGNAME, transparent = True)
        
    plt.show()
    plt.close()

#----------------------------------------------------------------------
# Main code (Call funcitons)

def main():
    sigmaZ_list = assembleZ_i()
    sigmaX_list = assembleX_i()

    Ising_Ham = assmemble_Hamiltonian(sigmaZ_list,sigmaX_list,H_Z,H_X)

    eigenvalues,eigenvectors = find_eigs(Ising_Ham)
    eigenvalue_spacings = find_spacings(eigenvalues)

    print(eigenvalue_spacings[0])
    spacings_hist(eigenvalue_spacings[1])
    #print_eigs(eigenvalues,eigenvectors)

#----------------------------------------------------------------------

if __name__ == "__main__":
    main()