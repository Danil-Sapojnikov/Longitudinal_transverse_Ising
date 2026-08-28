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
H_X = 0.5 # Transverse field
H_Z = 0.5 # Longitudinal field

NUM_EVALS = 3
TOL = 1e-10

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
    exchange += sigmaZ_list[i]@sigmaZ_list[i+1]
exchange = J * (exchange + sigmaZ_list[N-1]@sigmaZ_list[0])

transverse = sp.sparse.csr_array(np.zeros((2**N,2**N)))
for i in range(0,N):
    transverse += sigmaX_list[i]
transverse *= H_X

longitudinal = sp.sparse.csr_array(np.zeros((2**N,2**N)))
for i in range(0,N):
    longitudinal += sigmaX_list[i]
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

# Translates the energy eigenstates into the spin eigenstates that constitute it. (Made with ChatGPT)

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

# ============================================================
# Function to plot one eigenvector
# ============================================================

def plot_eigenvector(eigenvalue, eigenvector, eigenvector_number,
                     threshold=1e-8):

    # Find basis states that actually contribute
    amplitudes = np.asarray(eigenvector).flatten()

    #contributing = np.where(np.abs(amplitudes) > threshold)[0]
    contributing = np.where(np.abs(amplitudes)**2 > 0.01)[0]

    n_states = len(contributing)

    # One row for every contributing spin configuration
    fig, axes = plt.subplots(
        n_states,
        1,
        figsize=(8, max(2.5, 2.0 * n_states)),
        squeeze=False
    )

    axes = axes.flatten()

    fig.suptitle(
        f"Eigenvector {eigenvector_number+1}\n"
        f"$E = {eigenvalue.real:.6f}$",
        fontsize=16
    )

    # --------------------------------------------------------
    # Plot each contributing spin configuration
    # --------------------------------------------------------

    for row, basis_index in enumerate(contributing):

        ax = axes[row]

        spins = spin_configs[basis_index]
        amplitude = amplitudes[basis_index]

        # Real part -- your Hamiltonian is real
        amplitude = np.real(amplitude)

        # ----------------------------------------------------
        # Draw chain
        # ----------------------------------------------------

        x = np.arange(N)

        ax.plot(
            x,
            np.zeros(N),
            'k-',
            linewidth=1
        )

        # ----------------------------------------------------
        # Draw spins as arrows
        # ----------------------------------------------------

        for i, spin in enumerate(spins):

            if spin == 1:
                # Up spin
                ax.arrow(
                    i, 0,
                    0, 0.8,
                    head_width=0.12,
                    head_length=0.15,
                    length_includes_head=True
                )

                spin_label = r"$\uparrow$"

            else:
                # Down spin
                ax.arrow(
                    i, 0,
                    0, -0.8,
                    head_width=0.12,
                    head_length=0.15,
                    length_includes_head=True
                )

                spin_label = r"$\downarrow$"

            # Site number
            ax.text(
                i,
                1.05,
                f"{i+1}",
                ha='center',
                va='bottom',
                fontsize=11
            )

        # ----------------------------------------------------
        # Configuration label
        # ----------------------------------------------------

        config_string = ''.join(
            '↑' if s == 1 else '↓'
            for s in spins
        )

        ax.text(
            N + 0.3,
            0,
            rf"$|{config_string}\rangle$",
            va='center',
            fontsize=13
        )

        # ----------------------------------------------------
        # Amplitude and probability
        # ----------------------------------------------------

        probability = abs(amplitude)**2

        ax.text(
            -0.8,
            0,
            rf"$c={amplitude:.3f}$"
            "\n"
            rf"$P(s)={probability:.3f}$",
            ha='right',
            va='center',
            fontsize=11
        )

        # Formatting
        ax.set_xlim(-1.2, N + 1.5)
        ax.set_ylim(-1.3, 1.3)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Site {i+1}" for i in range(N)])

        ax.set_yticks([])

        # Remove unnecessary borders
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.show()


# ============================================================
# Plot every eigenvector
# ============================================================

for n in range(NUM_EVALS):

    plot_eigenvector(
        evals[n],
        evecs[:, n],
        n
    )
#----------------------------------------------------------------------
