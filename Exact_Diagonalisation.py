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

def poisson_dist(x):

    return np.exp(-x)

def wigner_dist(x):

    power = -np.pi / 4 * x**2

    return np.pi/2 * x * np.exp(power)

#----------------------------------------------------------------------
# Output functions

def print_eigs(evals,evecs):

    for i in range(len(evals)):
        print(f"\nEigenvalue {i+1}")
        print(f"{evals[i].real:.6f}")
        print(f"{np.round(np.real(evecs[:,i]),3)}")

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

def create_multiple_plots(plots, figsize=(12,8)):
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

        if "legend":
            ax.legend()


    for ax in axes[n_plots:]: 
        ax.set_visible(False) 

    fig.tight_layout() 

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
    scaled_spacings = np.delete(scaled_spacings, np.where(scaled_spacings > 10*np.mean(scaled_spacings))) #removes large values to siplify display

    ax.hist(scaled_spacings, bins=NUM_BINS, density = True, label = 'Calulated Differences')
    ax.plot(x,poisson_dist(x), label = 'Poisson Distribution', color = 'b')
    ax.plot(x,wigner_dist(x), label = 'GOE')

def assemble_spacings_plot_dict():
    return

#----------------------------------------------------------------------
# Main code (Call funcitons)

x = np.linspace(0,5,1000)

def main():
    sigmaZ_list = assembleZ_i()
    sigmaX_list = assembleX_i()

    Ising_Ham = assmemble_Hamiltonian(sigmaZ_list,sigmaX_list,H_Z,H_X)

    eigenvalues,eigenvectors = find_eigs(Ising_Ham)
    eigenvalue_spacings = find_spacings(eigenvalues)

    print(eigenvalue_spacings[0])
    #spacings_hist(eigenvalue_spacings[1])
    #print_eigs(eigenvalues,eigenvectors)

    plots_list = [{
        "plot": spacings_plot,
        "plotdata": (eigenvalue_spacings,np.linspace(0,5,1000)),
        "title": 'Energy Gap Probability Distribution',
        "xlabel": r'$\frac{s}{<s>}$',
        "ylabel": 'Probability Density',
        "legend": True
        },
        {"plot": spacings_plot,
        "plotdata": (eigenvalue_spacings,np.linspace(0,5,1000)),
        "title": 'Energy Gap Probability Distribution',
        "xlabel": r'$\frac{s}{<s>}$',
        "ylabel": 'Probability Density',
        "legend": False
        }]

    fig = create_multiple_plots(plots_list)
    if SAVEFIG:
        plt.savefig(FIGNAME, transparent = True) 
    plt.show()
    plt.close()

#----------------------------------------------------------------------

if __name__ == "__main__":
    main()