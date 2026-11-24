import numpy as np
import matplotlib.pyplot as plt
import task_2_2 as t

seeds = [1, 2, -1, -2]

sigmas_optimised, sigmas = t.optimise_sigma_for_each_n(seeds)
red_chi_sqs_by_sig= t.confirm_samples_for_each_n(sigmas_optimised, seeds)

'''
########
################
########################
################################
  !!! INCLUDE A PLOT HERE !!!
################################
########################
################
########
'''

# SAVE CONTENT FROM THIS CODE:

import os
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

np.savetxt(os.path.join(path, "sigmas_optimised.txt"), sigmas_optimised)
np.savetxt(os.path.join(path, "red_chi_sqs_by_sig.txt"), red_chi_sqs_by_sig)
np.savetxt(os.path.join(path, "sigmas.txt"), sigmas.reshape(sigmas.shape[0], -1))


# FOR LOADING CONTENT BACK INTO ORIGINAL FORM:

'''
import os
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

sigmas_optimised = np.loadtxt(os.path.join(path, "sigmas_optimised.txt"))
red_chi_sqs_by_sig = np.loadtxt(os.path.join(path, "red_chi_sqs_by_sig.txt"))
sigmas = np.loadtxt(os.path.join(path, "sigmas.txt")).reshape(-1, 2, 5)
'''
