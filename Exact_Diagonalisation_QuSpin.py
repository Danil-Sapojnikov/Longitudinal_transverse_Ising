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
import quspin

#----------------------------------------------------------------------

N = 12 # Number of elements in chain (N>=2)
J = 1.0 # Exchange coupling constant
H_X = 1.05 # Transverse field
H_Z = 0 # Longitudinal field

NUM_EVALS = 2**N - 1 #1000 # Number of Eigenvalues calculated
#TOL = 1e-10
NUM_BINS = 25

SAVETEXT = False
SAVEFIG = True
FIGNAME = 'N_12_AnaTest.png'
FIGTITLE = 'Comparison of the Numerical and Analytic solutions for the TFIM' #fr'Level spacings for the $h_x$ = {H_X} TFIM'

