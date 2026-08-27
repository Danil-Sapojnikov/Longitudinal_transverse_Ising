"""
***********************************************************************
* Exact Diagonalisation of the transverse field Ising model for low N. 

* Comparison to known analytic result.

* Inclusion of a longitudinal field.
***********************************************************************
"""

import numpy as np
import matplotlib as plt

#----------------------------------------------------------------------

N = 4 # Number of elements in chain
J = 1 # Exchange coupling constant
H_X = 0 # Transverse field
H_Z = 0 # Longitudinal field

SAVETEXT = False
SAVEFIG = False
#----------------------------------------------------------------------

def Ising_Hamiltonian():
    """ The Hamiltonian?! """

    