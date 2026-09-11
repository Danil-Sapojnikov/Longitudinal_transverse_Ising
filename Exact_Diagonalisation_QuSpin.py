"""
***********************************************************************
* Exact Diagonalisation of the transverse field Ising model for low N. 

* Comparison to known analytic result.

* Inclusion of a longitudinal field.

* Energy level spacings analysis.

* This verison utilises the QuSpin package.
***********************************************************************
"""

import numpy as np
import matplotlib.pyplot as plt
#import scipy as sp
#import quspin
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian

#----------------------------------------------------------------------

L = 4 # Number of elements in chain (L>=2)
J = 1.0 # Exchange coupling constant
H_X = 1.05 # Transverse field
H_Z = 0 # Longitudinal field

NUM_EVALS = 2**L - 1 #100 # Number of Eigenvalues calculated
TOL = 1e-12
NUM_BINS = 25

SAVETEXT = False
SAVEFIG = False
FIGNAME = 'L_12_AnaTest.png'
FIGTITLE = 'Comparison of the Numerical and Analytic solutions for the TFIM' #fr'Level spacings for the $h_x$ = {H_X} TFIM'

#----------------------------------------------------------------------
### Assemble the Longitudinal + Transverse field Ising Hamiltonian

def assemble_ising_hamiltonian(couplings, sites, periodic=True):
    """
    Assembles the longitudinal + transverse field Ising chain hamiltonian of the form:
    H = J*sum(sigmaZ_i sigmaZ_(i+1)) + hx*sum(sigmaX_i) + hz*sum(sigmaZ_i)

    Parameters
    ----------
    couplings: (Jzz,hx,hz) = tuple of floats.
    sites: number of spins in the Ising chain (Typically N or L).
    periodic: Boolean which determines if PBC are imposed on the chain (i.e. S_(N+1)= S_1 if True).

    Returns
    ----------
    Ising_Hamiltonian: QuSpin Hamiltonian quantum operator object. 
    """
    Jzz, hx, hz = couplings
    print(f"\nAssembling Hamiltonian with transverse field h_x = {hx} and longitudinal field h_z = {hz}.")
    basis = spin_basis_1d(L=sites)

    Jzz_list = [[Jzz, j,j+1] for j in range(sites-1)] # L-1 bonds
    if periodic:
        Jzz_list.append([Jzz, sites-1,0]) # Periodic B.C. wrap-around bond
    hx_list = [[hx,j] for j in range(sites)] # L sites transverse field coupling
    hz_list = [[hz,j] for j in range(sites)] # L sites longitudinal field coupling

    H_terms_static = [['zz',Jzz_list],
                      ['x',hx_list],
                      ['z',hz_list],
                      ]
    H_terms_dynamic = []

    Ising_Hamiltonian = hamiltonian(static_list=H_terms_static, dynamic_list=H_terms_dynamic, basis=basis)
    print("Hamiltonian assembled!")

    return Ising_Hamiltonian

def assemble_ising_ham_pk_blocks(couplings, sites, periodic=True):
    """
    Assembles the longitudinal + transverse field Ising chain hamiltonian of the form:
    H = J*sum(sigmaZ_i sigmaZ_(i+1)) + hx*sum(sigmaX_i) + hz*sum(sigmaZ_i)
    block-diagonalised into (p=+1(kstates),p=-1(kstates)) form.

    Parameters
    ----------
    couplings: (Jzz,hx,hz) = tuple of floats.
    sites: number of spins in the Ising chain (Typically N or L).
    periodic: Boolean which determines if PBC are imposed on the chain (i.e. S_(N+1)= S_1 if True).

    Returns
    ----------
    Ising_Hamiltonian: list of QuSpin Hamiltonian quantum operator objects. (block-diagonalised form of Ising_ham)
    """
    if not periodic:
        print("The Ising chain must be periodic or infinite to diagonalise it into k-blocks")
        return

    Jzz, hx, hz = couplings
    print(f"\nAssembling Hamiltonian with transverse field h_x = {hx} and longitudinal field h_z = {hz}.")

    kblocks = [i for i in range(sites)]
    Ham_list = []

    for parity in (+1,-1):
        for kblo in kblocks:
            symm_basis = spin_basis_1d(L=sites, a=1, kblock=kblo, pblock=parity)

#----------------------------------------------------------------------
### Find the Eigenvalues and Eigenvectors of the Hamiltonian, then the energy spacings

def find_eigs(Hamiltonian, sparse=False, num_eigs=1, return_evecs=False):
    """
    Uses sparse or dense methods to find some or all of the eigenvalues and eigenvectors of the input Hamiltonian.

    Parameters
    ----------
    Hamiltonian: A QuSpin Hamiltonian object
    sparse: Boolean, finds all the eigenvalues using numpy dense methods if False, only some using scipy.sparse methods if True
    num_eigs: Number of eigenvalues and eigenvectors to find using sparse methods
    return_evecs: If false then does not calculate the eigenvectors and returns an empty list for evecs

    Returns
    ----------
    evals, evecs: Numpy arrays of the eigenvalues and eigenvectors, with each eigenvector arranged in column form evecs[:,i]
    """
    evecs = []

    if sparse:
        print(f"\nFinding the first {num_eigs} eigenvalues using sparse methods.")
        if return_evecs:
            evals,evecs = Hamiltonian.eigsh(k=num_eigs, which='SA', tol=TOL, return_eigenvectors=return_evecs)
        else:
            evals = Hamiltonian.eigsh(k=num_eigs, which='SA', tol=TOL, return_eigenvectors=return_evecs)
    else:
        print(f"\nFinding all eigenvalues using dense methods.")
        if return_evecs:
            evals,evecs = Hamiltonian.eigh()
        else:
            evals = Hamiltonian.eigvalsh()

    print(f"{len(evals)} eigenvalues found.")

    return evals,evecs

def find_spacings(eigvals):
    """
    Finds the spacings between consecutive energy levels.

    Parameters
    ----------
    eigvals: numpy array of eigenvalues (within a given symmetry block for comparing integrability)

    Returns
    ----------
    eigvals_space, eigvals_space_scaled: numpy arrays of: the absolute gaps between consecutive energies, the gaps between consecutive energies divided by the mean spacing
    """
    eigvals = np.sort(eigvals)
    eigvals_shifted = np.insert(eigvals, 0, eigvals[0])
    eigvals_shifted = np.delete(eigvals_shifted, -1)

    eigvals_space = eigvals - eigvals_shifted

    mean = np.mean(eigvals_space)
    eigvals_space_scaled = eigvals_space/mean

    return eigvals_space, eigvals_space_scaled

def poisson_dist(x):

    return np.exp(-x)

def wigner_dist(x):

    power = -np.pi / 4 * x**2

    return np.pi/2 * x * np.exp(power)

#----------------------------------------------------------------------
### Analytic solution to the TFIM (h_z = 0) for benchmarking

def tfim_exact_energies(num, j, h_x):
    """
    Exact TFIM many-body spectrum for a periodic chain.

    Parameters
    ----------
    num : Number of spins. (Even)
    j : Ising coupling.
    h_x : Transverse field.
    parity : +1 or -1 

    Returns
    -------
    energies : numpy array, exact many-body energies in the relevant parity sector.
    """

    print(f"Calculating exact Eigenvalue spectrum for h_x = {h_x}")

    if num % 2 != 0:
        raise ValueError("N must be even.")

    energies = []

    for parity in (-1,+1):
        if parity == +1:
            kstates = (2*np.arange(num) + 1) * np.pi / num
        elif parity == -1:
            kstates = 2*np.arange(num) * np.pi / num

        if parity == +1: # The periodic fermion sector
            eps = 2*np.sqrt(j**2 + h_x**2 - 2*j*h_x*np.cos(kstates)) # Single-particle energies
            Egs = -0.5 * np.sum(eps) # Ground-state energy

            for state in range(2**num):
                occupation = np.array([(state >> i) & 1 for i in range(num)]) # Uses binary representations of numbers to create an array with the possible occupancies [e.g. 9 = 1001 = (1,0,0,1)]

                if (-1)**np.sum(occupation) == parity: # Only considers states with the correct parity
                    Energy = Egs + np.sum(occupation * eps)
                    energies.append(Energy)
        else: # The antiperiodic fermion sector requires special handling for k=0 and k=pi modes (these are unpaired and can reduce the energy instead of just increase it)
            special_states = [0,num//2] # These are the states where k=0 or k=pi.
            regular_states = [i for i in range(num) if i not in special_states]

            eps_regular = 2*np.sqrt(j**2 + h_x**2 - 2*j*h_x*np.cos(kstates[regular_states])) # Regular single-particle energies
            Azero = 2*(h_x - j) # Coefficients for the 0 and pi modes
            Api = 2*(h_x + j)
            Egs = -0.5 * np.sum(eps_regular) # Zero-occupation energy (Not necessarily ground state since the k=0 occupied state can sometimes lower the energy.)

            for state in range(2**num):
                occupation = np.array([(state >> i) & 1 for i in range(num)]) # Uses binary representations of numbers to create an array with the possible occupancies [e.g. 9 = 1001 = (1,0,0,1)]

                if (-1)**np.sum(occupation) == parity: # Only considers states with the correct parity

                    n0 = occupation[0] # Whether the spacial states are occupied or not (n0,npi = 0 or 1)
                    npi = occupation[num//2]

                    Energy = Egs + np.sum(occupation[regular_states] * eps_regular)
                    Energy += Azero * (n0 - 0.5)
                    Energy += Api * (npi - 0.5)
                    energies.append(Energy)
    print(f"{len(energies)} Eigenvalues calculated in Spectrum!")

    return np.sort(np.array(energies))

#----------------------------------------------------------------------
### Output functions

def create_multiple_plots(plots, figsize=(12,12),title=None):
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

    print(f"\nCreating figure.")
    n_plots = len(plots) 
    n_cols = 3 
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

def print_eigs(evals):

    for i in range(len(evals)):
        print(f"\nEigenvalue {i+1}")
        print(f"{evals[i].real:.6f}")

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
    max_dif = np.max(np.abs(num_data-ana_data))

    i = np.arange(len(num_data))

    ax.scatter(i,ana_data, marker = "s", color = 'y', label = 'Analytic eigenvalues')
    ax.scatter(i,num_data, marker = ".", color = 'b', label = 'Numerical eigenvalues')

    ax.legend(title = f"Maximum difference = {max_dif:.3}", loc='upper left')

def assemble_comparison_plot_dict(data,h_x,num):
    """
    Creates the required dictionary for the create_multiple_plots_function.

    Parameters
    ----------
    data: numpy array or tuple of arrays
    h_x: h_x value for title

    Returns
    ----------
    Dictionary
    """

    dict = {
        "plot": exact_comparison_plot,
        "plotdata": (data),
        "title": fr'$N$ = {num},   $h_x$ = {h_x}',
        "xlabel": 'Eigenvalue Number',
        "ylabel": 'Energy Eigenvalue',
    }
    return dict

#----------------------------------------------------------------------
### Main code (Call functions)

def main():

    #-----------------------------------------
    # Prepare empty lists and variables
    #-----------------------------------------

    H_X_list = [0, 0.3, 1.2]
    num_list = [4, 8]
    plots_list = []

    #-----------------------------------------
    # Compare eigenvalues with exact TFIM sol. for h_z = 0 
    #-----------------------------------------

    for num in num_list:

        for h_x in H_X_list:

            Ising_Ham = assemble_ising_hamiltonian(couplings=(J,h_x,0), sites=num)
            eigenvalues, eigenvectors = find_eigs(Hamiltonian=Ising_Ham)

            exact = tfim_exact_energies(num=num, j=J, h_x=h_x)[:len(eigenvalues)]

            plots_list.append(assemble_comparison_plot_dict((eigenvalues,exact),h_x,num))

    
    #------------------------------------------
    # Output text and generate figure
    #------------------------------------------ 
    
    fig = create_multiple_plots(plots_list,title = FIGTITLE)
    if SAVEFIG:
        plt.savefig(FIGNAME, transparent = True) 
    plt.show()
    plt.close()

#----------------------------------------------------------------------

if __name__ == "__main__":
    main()