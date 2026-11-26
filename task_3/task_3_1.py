# Define the new trial wavefunction
import numpy as np
import matplotlib.pyplot as plt

# Ill just make theta = pi for testing... NOTE this!!!

def psi(theta, r):
    return np.exp(-theta*r)

def cart_to_polar(r_vector):
    return np.sqrt((r_vector[0]**2 + r_vector[1]**2 + r_vector[2]**2))

# Now need to use a 3D matropolis sampler - for each x,y,z...
# -> I think need to be careful here - these points need to be independant of one another... This is something that I need to figure out in the code...
# * fundementally building this from the backbone of my task 2 code...

def Prob(theta, r):
    return np.abs(psi(theta, r))**2

# Linear congruential generator... (wikipedia table used to selct... https://en.wikipedia.org/wiki/Linear_congruential_generator)
def generate_cong(I_n, a = 1664525, c = 1013904223, m = 2**(32)):
    # I_n is the prev value
    # a is the multiplier
    # c is increment
    # m is the number of possible numbers

    I_n_1 = np.float64((a*(I_n) + c) % m)
    x_n_1 = I_n_1 / m

    return I_n_1, x_n_1

def guassian(x, mu, sigma, A=1): # Define A = 1 so can be directly compared to the probability...
    return A*np.exp(-(x-mu)**2/(2*sigma**2))

# Define a guassian with rejection sampling...
# Run a test on the sigma to determine optimum step size...

def sampling_guassian(I_prev, mu, sigma):
    # Define the range for the guassian over range_min_max
    half_range_min_max = 4*sigma

    I_u_new, u_i = generate_cong(I_prev)
    x_i = (mu - half_range_min_max) + (2*half_range_min_max*u_i) # Start from 4 sig out of the guassian's mu, then take the random selection to a point inside the 8sig range of the guassian...

    I_p_new, p_i = generate_cong(I_u_new) # p.d.f so can only be up to 1 - which it is...

    if p_i <= guassian(x_i, mu, sigma):
        return x_i, I_p_new

    else:
        # Recursively call the function again if it fails until we get a valid sample for the gaussian...
        return sampling_guassian(I_p_new, mu, sigma)


# NOTE: Now r_pos is a 3D vector (x,y,z) and so is sigma, and so is counts for each respective dimension...
def metropolis_step(theta, r_pos, I_prev, sigma, N_burn, total_counts = 0, accepted_counts = 0):
    r_pos_dash = np.zeros(3)
    for index, coord in enumerate(r_pos):
        r_pos_dash[index], I_prev = sampling_guassian(I_prev, mu=coord, sigma=sigma[index])

    P = Prob(theta, cart_to_polar(r_pos))
    P_dash = Prob(theta, cart_to_polar(r_pos_dash))

    frac = P_dash / P

    total_counts += 1

    if frac >= 1:
        r_pos = r_pos_dash
        if total_counts >= N_burn:
            accepted_counts += 1
        return r_pos, I_prev, total_counts, accepted_counts
    else:
        I_prev, u = generate_cong(I_prev)
        if u < frac:
            r_pos = r_pos_dash
            if total_counts >= N_burn:
                accepted_counts += 1
            return r_pos, I_prev, total_counts, accepted_counts
        else:
            r_pos = r_pos
            return r_pos, I_prev, total_counts, accepted_counts


def generate_N_metropolis_points(N, N_burn, sigma, I_prev, theta, r_pos0, total_counts = 0, accepted_counts = 0):
    samples = np.zeros((N, 3))
    r_pos = r_pos0

    if N_burn >= N:
        return "this is not gunna work"

    for point in range(0, N):
        r_pos, I_prev, total_counts, accepted_counts = metropolis_step(theta, r_pos, I_prev, sigma, N_burn, total_counts, accepted_counts)
        samples[point, :] = r_pos

    return samples[N_burn:], total_counts, accepted_counts # Clip the burn

# Note: N needs to be 20x larger than actual N as to avoid correlation we only take the 20th value every time...
def generate_samples(N_burn, sigma, theta, r_pos0, seeds, N=220000): # Function to generate independant samples
    total_counts = 0
    accepted_counts = 0
    samples = []

    for seed in seeds:
        sample, total_counts, accepted_counts = generate_N_metropolis_points(N=N, N_burn=N_burn, sigma=sigma, I_prev=seed, theta=theta, r_pos0=r_pos0, total_counts=total_counts, accepted_counts=accepted_counts)
        samples.append(sample[::20]) # Take every 20th sample for more non-correlated random samples

    samples = np.concatenate(samples)
    acceptance_rate_for_sigma_vector = (accepted_counts/(total_counts-N_burn))

    return samples, acceptance_rate_for_sigma_vector

#print(generate_samples(N_burn=2000, sigma=[1,2,1.5], theta=np.pi, r_pos0=[1,1,1], seeds=[1,2,-1,-2], N=220000))

# Load in the optimised sigmas
# _____________________________________________

import os
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

base_case_optimised_sigmas = np.loadtxt(os.path.join(path, "optimised_sigmas_task_3.txt"))

# Now I need to compare with the actual prob distribution
# _____________________________________________


def trap_intergration(rs, ys): # POINTS HAS FORM HERE x, y(x)
    h = (rs[1] - rs[0]) # Assume equal h
    return h*(-0.5*(ys[-1] + ys[0]) + np.sum(ys))

def check_samples_confirm(N_burn, sigma, theta, r_pos0, seeds):
    N_bins = 100
    bins = np.zeros((N_bins, 2))

    # if theta = 1 then we get basically all area between [0, 10]
    # However if theta = 0.05 then we need to increase upper limit to capture all area...
    upper_limit = 10/theta

    # Define centre point of each bin... Bounds from desmos...
    bins[:, 0] = np.linspace(0, upper_limit, N_bins) # Define bins in range of numbers...

    delta_r = np.abs((bins[1, 0] - bins[0, 0]))

    samples, _ = generate_samples(N_burn, sigma, theta, r_pos0, seeds)

    # Put number of values into bins...
    for pos in samples:
        val = np.sqrt(np.sum(pos**2))
        result = np.argmin(np.abs(bins[:, 0] - val))
        bins[result, 1] += 1

    # NOW NEED TO COMPARE SAMPLES TO THE EXPECTED DISTRIBUTION FROM rho(x):

    # Need to approximately normalise over a large range...
    # Choose a wide range to get 99.999% of area...
    rs = np.linspace(0, 10/theta, 1000)

    #IMPORTANT: To find the prob in a region (or the whole of r - which is what we tryin to do here) we need to intergrate over the entire 3D region in polar space...
    # So we do the surface area multipled by the psi**2 (prob) and then we intergrate over all of these shells w.r.t r...
    # This was missed initially and too a while to debug so mention in report???
    shell_areas = (4*np.pi*rs**2)*Prob(theta, rs)

    # Intergrate over the whole thing to produce a normalisation for P_x
    # HENCE: (produce norm const A)
    A = trap_intergration(rs, shell_areas)

    # print(A)

    rs = np.linspace(bins[0:, 0]-delta_r/2, bins[0:, 0]+delta_r/2, 20)
    shell_areas = (4*np.pi*rs**2)*Prob(theta, rs)

    prob = [trap_intergration(rs[:, i], shell_areas[:, i]) for i in range(N_bins)]

    # Now we can do P_x*delta_x = (P_x / A) * len(samples) = Expected val
    E = (prob / A) * len(samples)

    sigma_2 = E

    #print(E>=5) This makes a mask...
    mask = (E>=5) # Cauchran threashold
    #print(mask.sum(), N_bins)

    # WHAT IS THE 1 free prarm from the v in the red chi sq???

    red_chi_sq = (1/(mask.sum()-1))*np.sum(((bins[mask, 1] - E[mask])**2)/(sigma_2[mask]))

    return red_chi_sq

#theta = np.pi
# print(check_samples_confirm(N_burn=2000, sigma=base_caseoptimised_sigmas/theta, theta=theta, r_pos0=[1,2,1.5], seeds=[1,2,-1,-2,3,4,-4]))

# Now we need to create the function for doing the laplacian to then use in H to solve for E_l... 
# _____________________________________________

def laplacian(theta, x, y, z, h):

    # central difference 5 points

    wrt_x_plus_2h = psi(theta, cart_to_polar([x+2*h, y, z]))
    wrt_x_plus_h = psi(theta, cart_to_polar([x+h, y, z]))
    wrt_x_minus_h = psi(theta, cart_to_polar([x-h, y, z]))
    wrt_x_minus_2h = psi(theta, cart_to_polar([x-2*h, y, z]))

    wrt_y_plus_2h = psi(theta, cart_to_polar([x, y+2*h, z]))
    wrt_y_plus_h = psi(theta, cart_to_polar([x, y+h, z]))
    wrt_y_minus_h = psi(theta, cart_to_polar([x, y-h, z]))
    wrt_y_minus_2h = psi(theta, cart_to_polar([x, y-2*h, z]))

    wrt_z_plus_2h = psi(theta, cart_to_polar([x, y, z+2*h]))
    wrt_z_plus_h = psi(theta, cart_to_polar([x, y, z+h]))
    wrt_z_minus_h = psi(theta, cart_to_polar([x, y, z-h]))
    wrt_z_minus_2h = psi(theta, cart_to_polar([x, y, z-2*h]))

    y = psi(theta, cart_to_polar([x, y, z]))

    y_double_d_wrt_x = (-(15/6)*y + (4/3)*(wrt_x_plus_h+wrt_x_minus_h) - (1/12)*(wrt_x_plus_2h+wrt_x_minus_2h))/h**2
    y_double_d_wrt_y = (-(15/6)*y + (4/3)*(wrt_y_plus_h+wrt_y_minus_h) - (1/12)*(wrt_y_plus_2h+wrt_y_minus_2h))/h**2
    y_double_d_wrt_z = (-(15/6)*y + (4/3)*(wrt_z_plus_h+wrt_z_minus_h) - (1/12)*(wrt_z_plus_2h+wrt_z_minus_2h))/h**2

    return (y_double_d_wrt_x + y_double_d_wrt_y + y_double_d_wrt_z)


# Need to verify the implementation of the laplacian is correct... I guess just compare at say 100 points and calc the error from that?
# _____________________________________________

# Did proof on my ipad that: (ref in the paper): LAPLACIAN(e**(-theta*r)) = exp(-theta*r)[theta**2 - (2*theta)/r]

def check_lapacian(theta, x, y, z, h):
    r = cart_to_polar([x, y, z])
    expected_lap = np.float64(np.exp(-theta*r)*(theta**2 - (2*theta)/r))
    actual_lap = np.float64(laplacian(theta, x, y, z, h))

    return actual_lap, expected_lap

def check_lap_on_N_points(theta, h, optimised_sigmas):
    samples, _ = generate_samples(N_burn=2000, sigma=optimised_sigmas, theta=theta, r_pos0=[1,1,1], seeds=[1,2,-1,-2], N=220000)

    results = np.zeros(len(samples))

    for index, sample in enumerate(samples):
        acc, exp = check_lapacian(theta, sample[0], sample[1], sample[2], h)
        psi_val = np.float64(psi(theta, cart_to_polar(sample)))

        # Create the ratio in the difference from the abs(actual value - obtained value) / psi value - this should account for the difference for varying vals of psi causing differences in the like differences in error...
        results[index] = (np.abs(acc-exp)/psi_val)

    # Average over the ratio differences...
    avg_diff = (1/(len(results)))*np.sum(results)
    return avg_diff

h_optimised = 0.0026146732118010867

# print(check_lap_on_N_points(np.pi, h_optimised, optimised_sigmas/theta))
# We get a value of: 1.150587200192847e-07
# This is pretty good. However - its not as good as the difference in E terms we got in 2.1... This is most likely because for these wavefunctions there is like a maxima point which abruptly changes gradient at r=0... like if you think about this on a graph from r=0 to inf. then there is a discontinuous jump at the r=0 position... Hence we probs dont do as well when approxing this...

# Actually compute the E_l and then the average E with the error for a given theta...
# _____________________________________________

# Compute energies for a given theta - then we optimise for this later...

def compute_energies(theta, x, y, z, h, sigma):
    # Equation from script... Adjusted for E_l
    return ((-0.5*(1/(psi(theta, cart_to_polar([x, y, z]))))*laplacian(theta, x, y, z, h)) - (1 / (cart_to_polar([x, y, z]))))


def avg_energy_given_theta(theta, h, sigma, seeds, N_burn, r_pos0, N = 220000):
    samples, _ = generate_samples(N_burn=N_burn, sigma=sigma, theta=theta, r_pos0=r_pos0, seeds=seeds, N=N)
    results = np.zeros(len(samples))

    for index, sample in enumerate(samples):
        results[index] = compute_energies(theta, sample[0], sample[1], sample[2], h, sigma)

    mean_energy = (1/(len(results)))*np.sum(results)
    sample_varience = (1/(len(results)-1))*(np.sum((results - mean_energy)**2))
    varience_of_mean = (1/len(results))*(sample_varience)

    return mean_energy, np.sqrt(varience_of_mean)

# print(avg_energy_given_theta(1, h_optimised, optimised_sigmas/theta))


# Now need to do gradient decent (from fig at bottom) to update thet theta so...
# theta_n+1 = theta_n - alpha*(d/dtheta * <H>) <- given in chap 12
# Alpha is step size...
# we basically keep going until theta_n+1 -> theta_n (converges)
# gunaa reccord theta_n and <H> for each theta... for each one...
# _____________________________________________

def compute_expected_hamiltonian_derivitive_wrt_theta(theta, h, sigma, seeds, N_burn, r_pos0, N = 220000):
    # We know psi = e**(-theta*r)
    # Hence d*psi/d_theta = -r*e**(-theta*r) // This could be done numerically - but that introducs errors so I think i will do the obvious form here as this will be exact??? Ask in seminar if i can just do this?

    # Because we use the sample seed - this will compute deterministicly the exactly same results for E_l we used to make the <H>...
    samples, _ = generate_samples(N_burn=N_burn, sigma=sigma, theta=theta, r_pos0=r_pos0, seeds=seeds, N=N)

    N_s = len(samples)

    E_l = compute_energies(theta=theta, x=samples[:, 0], y=samples[:, 1], z = samples[:, 2], h=h, sigma=sigma)

    # This is stupid - to call this function below I would need to resample the same seed again - which is just doubling the metropolis sampling requifrements to run this...
    #exp_H, exp_H_std = avg_energy_given_theta(theta, h, sigma, seeds, N_burn, r_pos0, N)
    # Calc it again just raw here from E_l
    exp_H = 1/(len(E_l))*np.sum(E_l)

    psi_for_rs = np.exp(-1*cart_to_polar([samples[:, 0], samples[:, 1], samples[:, 2]])*theta)

    d_wrt_theta_of_psi_over_psi = -cart_to_polar([samples[:, 0], samples[:, 1], samples[:, 2]]) # * psi_for_rs/psi_for_rs this could have just cancel from the very start!!!

    d_exp_H_by_d_theta = (2/N_s)*np.sum((E_l - exp_H)*d_wrt_theta_of_psi_over_psi)
    return d_exp_H_by_d_theta, exp_H

def grad_decent_step(theta_n, expected_hamiltonian_wrt_theta, alpha):
    return (theta_n - alpha*expected_hamiltonian_wrt_theta)

def grad_decent(alpha, max_number_of_steps, error_tolerence, initial_theta, h, sigma, seeds, N_burn, r_pos0, N = 220000):
    theta_n_1 = np.inf
    theta_n = initial_theta
    count = 0
    results = np.zeros((max_number_of_steps, 2)) # Store theta_n and Exp_at_theta_n

    while (np.abs(theta_n_1-theta_n) >= error_tolerence):
        if count != 0:
            theta_n = theta_n_1

        if count >= max_number_of_steps:
            return results, False

        # REMEMBER! We scale the base_case optimised sigmas to the specific theta we have... So if theta is larger then we need a smaller step size theta! Hence the div by theta... Look at the optimise_sig for understanding when writing report - initially made mistake of doing times...!
        d_exp_H_by_d_theta, exp_H = compute_expected_hamiltonian_derivitive_wrt_theta(theta_n, h, sigma/theta_n, seeds, N_burn, r_pos0, N)
        theta_n_1 = grad_decent_step(theta_n, d_exp_H_by_d_theta, alpha)
        results[count, 0] = theta_n
        results[count, 1] = exp_H
        
        count += 1

    # Need to mask the result array incase it converges in less than max_number_of_steps...
    mask = results[:, 0] > 0
    return results[mask, :], True

_, initial_theta = generate_cong(327)
initial_theta = 2*np.pi*initial_theta

print("initial theta is chosen between 0 and 2pi which is chosen to be:", initial_theta)

results, converged = grad_decent(alpha=0.1, max_number_of_steps = 100, error_tolerence=10**(-4), initial_theta=initial_theta, h=h_optimised, sigma=base_case_optimised_sigmas, seeds=[1, 2, -1, -2], N_burn=2000, r_pos0=[1, 2, 1.5], N = 22000)

'''
plt.plot(results[:, 0], results[:, 1])
plt.show()

print("Converged? :",converged, "|||| Converged in", len(results), "steps ||||", "converged at theta =", results[-1, 0], "for energy", results[-1, 1])
'''

# Now just take this optimum theta and do a full formal run with like 30k points?
# _____________________________________________

optimal_theta = results[-1, 0]
seeds=[1, 2, -1, -2, -3, 3, -4] # 30k runs across 7 seeds...

'''
best_energy, best_std = avg_energy_given_theta(theta=optimal_theta, h=h_optimised, sigma=base_case_optimised_sigmas, seeds=seeds, N_burn=2000, r_pos0 = [1, 2, -1.5], N = int((30000*20/(len(seeds)))))

print("best energy and std:", best_energy, "+/-", best_std)
if (np.abs(-0.5 - best_energy) <= best_std):
    print("the result should be -0.5 (lab script ref) - so we ARE in error range!!!")
'''
# Why is it neccicery to do this:
# -> In the earlier method - it would take to long if i did 30k iteratins for each theta and then created a new sigma from that...
# -> These energies I got were good but less accurate estimates (not as many iterations means the metropolis hasnt had time to fully tend toward a red chi squared of about 1 when compared to the actual pdf distribution - so a bit noiseier...)... Running it over the full 30k iterations for an optimised theta value gave a accurate result...



# Now just need to produce the density plot
# _____________________________________________
# For ground state we just used the minimised theta which we just calculated for...


optimal_theta = results[-1, 0]
seeds=[1, 2, -1, -2, -3, 3, -4] # 0.3M runs across 7 seeds...

# Generate 0.3M samples for the plot - this is across 500 bins - so will be high resolution...
samples, _ = generate_samples(N_burn=2000, sigma=base_case_optimised_sigmas/optimal_theta, theta=optimal_theta, r_pos0=[1, 1.5, 2], seeds = seeds, N=int(3000000*20/len(seeds)))

# Plot with a histogram...

samples_x, samples_y = samples[:, 0], samples[:, 1]

#matplotlib.pyplot.hist2d(x, y, bins=10, *, range=None, density=False, weights=None, cmin=None, cmax=None, data=None, **kwargs)

plt.figure()
plt.hist2d(samples_x, samples_y, bins=300, range=[[-2.5, 2.5],[-2.5, 2.5]])
plt.gca().set_aspect("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Ground state probability density of the wavefunction (projected on x-y plane)\nFor minimised theta = {optimal_theta:.4f}")
plt.show()