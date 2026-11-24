import numpy as np
from task_3_1 import generate_samples

# Now I need to optimise for the best sigmas...
# _____________________________________________

def optimise_sigma(N_burn, theta, r_pos0, seeds):
    theta = 1 # Normalise to theta = 1 here then scale - this later...
    # If we do the standard_optimal_sigma/theta then we scale the best sigma... I.e if theta = 2pi over pi then the function decays twice as fast... 
    # This means greater rate of change - so the sigma needs to be smaller - hence steps are smaller too!
    # This means we can scale later depending on problem...

    sigma = np.linspace(0.5, 3, 50) / theta
    acceptance_rates = np.zeros((len(sigma), 4))

    for index, sig in enumerate(sigma):
        # Here the function is actually symetric in r so the sigma should be the same for all three directions...
        _, acceptance_rate_for_sigma_vector = generate_samples(N_burn, [sig, sig, sig], theta, r_pos0, seeds, N=22000)
        acceptance_rates[index, 0] = (acceptance_rate_for_sigma_vector)
        acceptance_rates[index, 1], acceptance_rates[index, 2], acceptance_rates[index, 3] = sig, sig, sig

    # In for high dimensional ND the optimal acceptance rate is 0.234 (source: gelman, robers, gilts... 1997)
    return acceptance_rates[np.argmin(np.abs(acceptance_rates[:, 0] - 0.234)), 1:4]

optimised_sigmas_task_3 = optimise_sigma(2000, np.pi, [1, 2, 1.5], [1, -1, 2, -2])

import os
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
np.savetxt(os.path.join(path, "optimised_sigmas_task_3.txt"), optimised_sigmas_task_3)
