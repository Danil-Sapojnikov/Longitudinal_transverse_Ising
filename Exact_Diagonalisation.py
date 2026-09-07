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

N = 10 # Number of elements in chain (N>=2)
J = 1 # Exchange coupling constant
H_X = 1.05 # Transverse field
H_Z = 0 # Longitudinal field

NUM_EVALS = 1000 # Number of Eigenvalues calculated
#TOL = 1e-10
NUM_BINS = 25

SAVETEXT = False
SAVEFIG = False
FIGNAME = 'EigenvalueSpacings2.png'
FIGTITLE = 'Comparison of the Numerical and Analytic solutions for the TFIM' #fr'Level spacings for the $h_x$ = {H_X} TFIM'

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

    print(f"\nAssembling Hamiltonian with transverse field h_x = {h_x} and longitudinal field h_z = {h_z}.")

    exchange = sp.sparse.csr_array(np.zeros((2**N,2**N)))
    for i in range(0,N-1):
        exchange += sigmaZ_list[i]@sigmaZ_list[i+1]
    exchange = J * (exchange + sigmaZ_list[N-1]@sigmaZ_list[0]) # Introduces periodic boundary conditions
    #exchange = J * (exchange + sigmaZ_list[N-1]@I) # Open Boundary conditions?

    transverse = sp.sparse.csr_array(np.zeros((2**N,2**N)))
    for i in range(0,N):
        transverse += sigmaX_list[i]
    transverse *= h_x

    longitudinal = sp.sparse.csr_array(np.zeros((2**N,2**N)))
    for i in range(0,N):
        longitudinal += sigmaZ_list[i]
    longitudinal *= h_z

    Hamiltonian = -exchange - transverse - longitudinal

    print("Hamiltonian assembled.")
    #print(Hamiltonian.toarray())
    return Hamiltonian

#----------------------------------------------------------------------
# Find the Eigenvalues and Eigenvectors of the Hamiltonian, then the energy spacings

def find_eigs(Hamiltonian):

    print(f"\nFinding energy eigenvalues.")
    #evals,evecs = sp.sparse.linalg.eigs(Hamiltonian, k = NUM_EVALS, which = 'SR')
    evals,evecs = sp.sparse.linalg.eigsh(Hamiltonian, k = NUM_EVALS, which = 'SA')
    print(f"{len(evals)} out of {NUM_EVALS} eigenvalues found.")
    #print(f"{len(np.real_if_close(evals[~np.isnan(evals)],tol=100))} out of {len(evals)} eigenvalues are real numbers.")

    return evals,evecs

def find_spacings(eigvals):

    eigvals = np.sort(eigvals)
    eigvals_shifted = np.insert(eigvals, 0, eigvals[0])
    eigvals_shifted = np.delete(eigvals_shifted, -1)

    eigvals_space = eigvals - eigvals_shifted

    mean = np.mean(eigvals_space)
    eigvals_space_scaled = eigvals_space/mean

    return (eigvals_space,eigvals_space_scaled)

def poisson_dist(x):

    return np.exp(-x)

def wigner_dist(x):

    power = -np.pi / 4 * x**2

    return np.pi/2 * x * np.exp(power)

#----------------------------------------------------------------------
# Analytic TFIM (h_z = 0)

def tfim_exact_energies(h_x):
    """
    Exact TFIM many-body spectrum for a periodic chain.

    Parameters
    ----------
    N : Number of spins. (Even)
    J : Ising coupling.
    h_x : Transverse field.
    parity : +1 or -1 

    Returns
    -------
    energies : numpy array, exact many-body energies in the relevant parity sector.
    """

    if N % 2 != 0:
        raise ValueError("N must be even.")

    energies = []

    for parity in (-1,+1):
        if parity == +1:
            ks = (2*np.arange(N) + 1) * np.pi / N
        elif parity == -1:
            ks = 2*np.arange(N) * np.pi / N
        else:
            raise ValueError("parity must be +1 or -1")

        eps = 2*np.sqrt(J**2 + h_x**2 - 2*J*h_x*np.cos(ks)) # Single-particle energies
        E0 = -0.5 * np.sum(eps) # Ground-state energy

        for state in range(2**N):
            occupation = np.array(
                [(state >> k) & 1 for k in range(N)]
            ) # Uses binary representations of numbers to create an array with the possible occupancies [e.g. 9 = 1001 = (1,0,0,1)]

            if (-1)**np.sum(occupation) == parity: # Only considers states with the correct parity
                E = E0 + np.sum(occupation * eps)
                energies.append(E)

    return np.sort(np.array(energies))

#----------------------------------------------------------------------
# Output functions

def print_eigs(evals,evecs):

    for i in range(len(evals)):
        print(f"\nEigenvalue {i+1}")
        print(f"{evals[i].real:.6f}")
        #print(f"{np.round(np.real(evecs[:,i]),3)}")

def create_multiple_plots(plots, figsize=(12,8),title=None):
    """ 
    Creates a figure with a variable number of subplots. [Created with the aid of ChatGPT]

    Parameters 
    ---------- 
    plots : list of dict Each dictionary describes one subplot. It should contain: 
        - "plot": a function that accepts an Axes object and the data for plots/scatters/hist
        - "plotdata": a (linspaced) numpy array that is used as the x values or a tuple of arrays, e.g (histdata,xdata,ydata,...) for a scatter plot or multiple subplots
        - "title": optional title for the subplot 
        - "xlabel": optional x-axis label 
        - "ylabel": optional y-axis label 
        - "legend": optional subplot legend True or False
    
    figsize : tuple 
        Size of the overall figure. 
    
    Returns 
    ------- 
    fig : matplotlib.figure.Figure 
    """

    print(f"Creating figure.")
    n_plots = len(plots) 
    n_cols = 2 
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for ax, plot_info in zip(axes, plots): 
        plot_info["plot"](ax,plot_info["plotdata"])

        if "title" in plot_info:
            ax.set_title(plot_info["title"])

        if "xlabel" in plot_info:
            ax.set_xlabel(plot_info["xlabel"])
 
        if "ylabel" in plot_info:
            ax.set_ylabel(plot_info["ylabel"])


    for ax in axes[n_plots:]: 
        ax.set_visible(False) 

    if title:
        fig.suptitle(title, fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.98]) 

    return fig

def spacings_plot(ax,plot_data):
    """
    Plots a histogram of the spacings data with a Poisson and Wigner distribution for comparison.

    Parameters
    ----------
    ax: Axes object to plot the graph on
    plot_data: tuple of data (histdata,x) where x is a linspaced array for the distributions.
    """
    scaled_spacings,x = plot_data
    scaled_spacings = np.delete(scaled_spacings, np.where(scaled_spacings > 7*np.mean(scaled_spacings))) #removes large values to siplify display

    ax.hist(scaled_spacings, bins=NUM_BINS, density = True, label = 'Calulated Differences')
    ax.plot(x,poisson_dist(x), label = 'Poisson Distribution', color = 'b')
    ax.plot(x,wigner_dist(x), label = 'GOE')
    ax.legend()

def assemble_spacings_plot_dict(data,h_z):
    """
    Creates the required dictionary for the create_multiple_plots_function.

    Parameters
    ----------
    data: numpy array or tuple of arrays
    h_z: h_z value for title

    Returns
    ----------
    Dictionary
    """

    plot_dict = {
        "plot": spacings_plot,
        "plotdata": data,
        "title": fr'Energy Gap Probability Distribution for $h_z$ = {h_z}',
        "xlabel": r'$\frac{s}{<s>}$',
        "ylabel": 'Probability Density',
    }

    return plot_dict

def exact_comparison_plot(ax,plot_data):
    """
    Plots two scatter plots of the exact tfim against the calculated one.

    Parameters
    ----------
    ax: Axes object to plot the graph on
    plot_data: tuple of data (num_data,ana_data).
    """
    num_data,ana_data = plot_data

    i = np.arange(len(num_data))

    ax.scatter(i,ana_data, marker = "s", color = 'y', label = 'Analytically calculated eigenvalues')
    ax.scatter(i,num_data, marker = "x", color = 'b', label = 'Numerically solved eigenvalues')

    ax.legend()
#----------------------------------------------------------------------
# Main code (Call funcitons)

def main():
    sigmaZ_list = assembleZ_i()
    sigmaX_list = assembleX_i()

    H_Z_list = [0.1,0.2,0.3,0.4] #[0,0.2,0.4,0.6,0.8,1.0]
    plots_list = []


    #for i in range(len(H_Z_list)):

    #    Ising_Ham = assmemble_Hamiltonian(sigmaZ_list,sigmaX_list,H_Z_list[i],H_X)

    #    eigenvalues,eigenvectors = find_eigs(Ising_Ham)
    #    eigenvalue_spacings = find_spacings(eigenvalues)

    #    x = np.linspace(0,7,1000)

    #    eigenvalue_spacings_real = np.real_if_close(eigenvalue_spacings,tol=100)
    #    plots_list.append(assemble_spacings_plot_dict((eigenvalue_spacings_real[1],x),H_Z_list[i]))


    Ising_Ham = assmemble_Hamiltonian(sigmaZ_list,sigmaX_list,H_Z,H_X)
    eigenvalues,eigenvectors = find_eigs(Ising_Ham)

    #print(eigenvalue_spacings[0])
    #print(eigenvalues)
    exact = tfim_exact_energies(H_X)[:NUM_EVALS]
    dif = eigenvalues-exact
    print(f"Maximum deviation: {np.max(dif)}")

    plots_list = [{
        "plot": exact_comparison_plot,
        "plotdata": (eigenvalues,exact),
        "title": fr'$h_x$ = {H_X}',
        "xlabel": 'Eigenvalue Number',
        "ylabel": 'Energy',
    }]
    
    fig = create_multiple_plots(plots_list,title = FIGTITLE)
    if SAVEFIG:
        plt.savefig(FIGNAME, transparent = True) 
    plt.show()
    plt.close()

#----------------------------------------------------------------------

if __name__ == "__main__":
    main()