from task_2_2 import find_mean_and_std
import numpy as np

# LOAD IN CONTENT:
# FOR LOADING CONTENT BACK INTO ORIGINAL FORM:

import os
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

sigmas_optimised = np.loadtxt(os.path.join(path, "sigmas_optimised.txt"))
red_chi_sqs_by_sig = np.loadtxt(os.path.join(path, "red_chi_sqs_by_sig.txt"))
sigmas = np.loadtxt(os.path.join(path, "sigmas.txt")).reshape(-1, 2, 5)

# Finish the next 2 bullet points...

# This is for the 5point - this was found in task_2_1
h_optimised = 0.0026146732118010867 
sigmas_opt = np.zeros((len(sigmas_optimised), 2))
sigmas_opt[:, 0] = sigmas_optimised

Es = np.zeros((5, 3))

for n in [0, 1, 2, 3, 4]:
    Es[n, :] = find_mean_and_std(sigmas_opt, n, h_optimised)
    print("ACTUAL:",Es[n, 0], "    OBTAINED:", Es[n,1], "    ERROR:", Es[n, 2])

    if (np.abs(Es[n, 2]) >= np.abs(np.float64(Es[n, 0]-Es[n, 1]))):
        print("Inside error bounds! (by: ", np.abs(Es[n, 2] - np.float64(Es[n, 0]-Es[n, 1])), ")")

print(Es)
