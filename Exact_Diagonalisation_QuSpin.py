"""
***********************************************************************
* Exact Diagonalisation of the transverse field Ising model for low N. 

* Comparison to known analytic result.

* Inclusion of a longitudinal field.

* Energy level spacings analysis.

* This verison utilises the QuSpin package.
***********************************************************************
"""

# check quspin installation
import sys, os
#sys.path.append("/usr/local/lib/python3.10/site-packages") # path to local installation in google drive

import quspin
#print('\nquspin {} installed successfully and ready for use!\n'.format(quspin.__version__) )

# set options and import required packages
os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='4' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel

# import plotting tool
from matplotlib import pyplot as plt
plt.rcParams.update({
    "text.usetex": True, # enable latex font
    "font.family": "Helvetica", # set font style
    "text.latex.preamble": r'\usepackage{amsmath}', # add latex packages
    "font.size": "16", # set font size
})
#%matplotlib inline

# import numpy
import numpy as np
# suppress machine precision; print numbers up to three decimals
np.set_printoptions(suppress=True,precision=3)