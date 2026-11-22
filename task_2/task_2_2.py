# TRANSFORMTAION METHOD WONT WORK (past n =1) - it is simple to invert for the exponential alone, but any further than that (n=1,2,3,4... cases cant work) and basically impos
# REJECTION SAMPLING: This could work, could define region out to very large x and y, however as we increase n to larger and larger behavior and probability distribution approx will get worse...
# METROPOLIS SAMPLING: This could work better i think, we don't need to worry about the normalising of pdf - this is a benifit as it basically spreads over infinity and we make approx as with the rejection method if we dont intergrate over all bounds...

import numpy as np
seeds = [1, 2, -1, -2]

def hermite(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return 2*x
    elif n == 2:
        return 4*x**2 - 2
    elif n == 3:
        return 8*x**3 - 12*x
    elif n == 4:
        return 16*x**4 - 48*x**2 + 12
    else:
        return 0 # This should never be called...

def psi(n, x):
    return hermite(n, x)*(np.exp(-x**2/2))

# Algorithm for metropolis sampling
# -> Generates samples with density proportional to rho_x
# -> NOTE: rho_x is the normalised probability over the whole range... This is literally unobtainable as discussed above - include this in my report!
# -> NOTE: So i think we can use the fact from QM that Prob = mod(psi)**2 - this is not normalised here but we divide new by old (REF NOTES HERE) so the normalisation would cancel anyway...
# |--> P_x = np.abs(psi(x))**2

def P_x(n, x):
    return np.abs(psi(n, x))**2

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

def metropolis_step(n, x, I_prev, sigma, N_burn, total_counts = 0, accepted_counts = 0):
    x_dash, I_x_n = sampling_guassian(I_prev, mu=x, sigma=sigma)

    P = P_x(n, x)
    P_dash = P_x(n, x_dash)

    frac = P_dash / P

    total_counts += 1

    if frac >= 1:
        x = x_dash
        if total_counts >= N_burn:
            accepted_counts += 1
        return x, I_x_n, total_counts, accepted_counts
    else:
        I_u_n, u = generate_cong(I_x_n)
        if u < frac:
            x = x_dash
            if total_counts >= N_burn:
                accepted_counts += 1
            return x, I_u_n, total_counts, accepted_counts
        else:
            x = x
            return x, I_u_n, total_counts, accepted_counts


def trap_intergration(xs, ys): # POINTS HAS FORM HERE x, y(x)
    h = (xs[1] - xs[0]) # Assume equal h
    return h*(-0.5*(ys[-1] + ys[0]) + np.sum(ys))

def generate_N_metropolis_points(N, N_burn, sigma, n, x0, I_prev, total_counts = 0, accepted_counts = 0):
    x_samples = np.zeros((N))
    x = x0

    if N_burn >= N:
        return "this is not gunna work"

    for point in range(0, N):

        x, I_new, total_counts, accepted_counts = metropolis_step(n, x, I_prev, sigma, N_burn, total_counts, accepted_counts)
        I_prev = I_new
        x_samples[point] = x

    return x_samples[N_burn:], total_counts, accepted_counts # Clip the burn

'''
sol = (generate_N_metropolis_points(10000, 2000, 2, n=2, x0 = 3))
max = np.argmax(sol)
min = np.argmin(sol)
print(sol[max], sol[min])

    
'''

# Note: N needs to be 20x larger than actual N as to avoid correlation we only take the 20th value every time...
def generate_samples(N_burn, index, sigma, n, x0, seeds, N=220000): # Function to generate independant samples
    total_counts = 0
    accepted_counts = 0
    samples = []

    for seed in seeds:
        sample, total_counts, accepted_counts = generate_N_metropolis_points(N, N_burn=N_burn, sigma=sigma[index, 0], n=n, x0 = x0, I_prev=seed, total_counts=total_counts, accepted_counts=accepted_counts)
        samples.append(sample[::20]) # Take every 20th sample for more non-correlated random samples

    sigma[index, 1] = (accepted_counts/(total_counts-N_burn))
    samples = np.concatenate(samples)

    return samples, sigma


def optimise_sigma(N_burn, n, x0, seeds):
    sigma = np.zeros((50, 2))
    sigma[:, 0] = np.linspace(0.5, 5, 50) # 50 points for sigma between 0.5 and 5...
    
    for index in range(0, len(sigma[:, 0])):
        _, sigma = generate_samples(N_burn, index, sigma, n, x0, seeds)
    # We now have complete dataset for sigma...

    # Return the value of sigma with the acceptance rate closest to 44%
    return sigma[np.argmin(np.abs(sigma[:, 1] - 0.44)), 0], sigma


def check_samples_confirm(N_burn, sigma_single, n, x0, seeds):
    N_bins = 100
    bins = np.zeros((N_bins, 2))

    # Define centre point of each bin... Bounds from desmos...
    bins[:, 0] = np.linspace(-6, 6, N_bins) # Define bins in range of numbers...

    delta_x = np.abs((bins[1, 0] - bins[0, 0]))

    # sigma is a 2D array with dimensions (1,2) - so index = 0 when calling generate samples here...
    sigma = np.zeros((1, 2))
    sigma[0, 0] = sigma_single
    samples, _ = generate_samples(N_burn=N_burn, index=0, sigma=sigma, n=n, x0=x0, seeds=seeds)

    # Put number of values into bins...
    for val in samples:
        result = np.argmin(np.abs(bins[:, 0] - val))
        bins[result, 1] += 1

    # NOW NEED TO COMPARE SAMPLES TO THE EXPECTED DISTRIBUTION FROM rho(x):

    # Need to approximately normalise over a large range...
    # Choose a wide range to get 99.999% of area...
    xs = np.linspace(-50, 50, 1000)
    ys = P_x(n=n, x = xs)

    # Intergrate over the whole thing to produce a normalisation for P_x
    # HENCE: (produce norm const A)
    A = trap_intergration(xs, ys)

    # print(A)

    xs = np.linspace(bins[0:, 0]-delta_x/2, bins[0:, 0]+delta_x/2, 20)
    ys = P_x(n=n, x = xs)

    prob = [trap_intergration(xs[:, i], ys[:, i]) for i in range(N_bins)]

    # Now we can do P_x*delta_x = (P_x / A) * len(samples) = Expected val
    E = (prob / A) * len(samples)

    '''
    WHY SIGMA_2 = E

    Think of each sample as a coin flip: it either lands in your bin, with probability p, or it doesn’t. Over N samples, the count follows a binomial distribution, and its variance is N p times one minus p. Your bins are narrow, so p is small, and one minus p is basically one. That leaves N times p, which is exactly your expected count E. So the variance equals E. That’s the Poisson limit.
    '''

    sigma_2 = E

    '''
    First, the chi-squared formula assumes each bin count has a roughly bell-shaped spread around E. That holds when E is reasonably big, but with tiny E it’s lopsided. Say E is 0.01 out in the tail and one sample happens to land there: that bin alone adds about 100 to your sum, because you’re dividing by 0.01. A single fluke can swamp everything, so those bins aren’t trustworthy.

    Second, the opposite case: tail bins where E is basically zero and nothing lands. They add almost nothing to the sum, but each still counts as a degree of freedom. So you’re dividing by a bigger number for no reason, which drags chi-squared down and makes the sampler look better than it is. That’s why you went from 1.38 to 2.33 with the filter.

    The number 5 is popularised by statistician cochran in 50's

    '''

    #print(E>=5) This makes a mask...
    max_count_in_a_bin = bins[[np.argmax(bins[:, 1])], 1].item()
    mask = (E>=5) # Cauchran threashold
    #print(mask.sum(), N_bins)

    # WHAT IS THE 1 free prarm from the v in the red chi sq???

    red_chi_sq = (1/(mask.sum()-1))*np.sum(((bins[mask, 1] - E[mask])**2)/(sigma_2[mask]))

    return red_chi_sq


def confirm_samples_for_each_n(sigmas_optimised, seeds):
    red_chi_sqs_by_sig = np.zeros((5, 3))
    red_chi_sqs_by_sig[:, 0] = np.array([0, 1, 2, 3, 4])
    red_chi_sqs_by_sig[:, 1] = sigmas_optimised
    for n in [0,1,2,3,4]:
        red_chi_sqs_by_sig[n, 2] = check_samples_confirm(N_burn=2000, sigma_single=sigmas_optimised[n], n=n, x0=1.5, seeds=seeds)
        print("For the n="+str(n),"we get the red chi squared to be:",red_chi_sqs_by_sig[n, 2])

    return red_chi_sqs_by_sig

def optimise_sigma_for_each_n(seeds):
    sigmas_optimised = np.zeros((5))
    sigmas = np.zeros((50, 2, 5))
    for n in [0,1,2,3,4]:
        sigmas_optimised[n], sigma = optimise_sigma(N_burn=2000, n=n, x0=1.5, seeds=seeds)
        print("For the n="+str(n),"we get the optimised sigma to be:",sigmas_optimised[n])
        sigmas[:, :, n] = sigma
    return sigmas_optimised, sigmas



# NOTE: The last bit of the first bullet point of 2.2 is moved into task_2_2_ext.py as it would cause code to run for like 30 mins...

def generateData(sigmas_optimised, n, no_of_points=10000):
    points = np.zeros((no_of_points))
    points, _ = generate_samples(N_burn=2000, index=n, sigma=sigmas_optimised, n=n, x0=1.5, seeds = seeds, N=(int(no_of_points*20/4))) # NOTE: x20 as we only sample every 20 points, x4 as we have 4 seperate seeds which we iterate through as well
    return points

def central_difference_5_points(n, x, h):
    y_double_d = np.float64(np.zeros(len(x)))
    y_plus_2h = psi(n, x+2*h)
    y_plus_h = psi(n, x+h)
    y_minus_h = psi(n, x-h)
    y_minus_2h = psi(n, x-2*h)
    y = psi(n, x)

    y_double_d = (-(15/6)*y + (4/3)*(y_plus_h+y_minus_h) - (1/12)*(y_plus_2h+y_minus_2h))/h**2

    return y_double_d

def find_local_energies(sigmas_optimised, n, h):
    points = generateData(sigmas_optimised, n)
    double_d = central_difference_5_points(n, points, h)
    local_Es = (-1/2) * (1/psi(n, points)) * double_d + (1/2) * (points)**2
    return local_Es

def average_energies(sigmas_optimised, n, h):
    local_Es = find_local_energies(sigmas_optimised, n, h)
    E_expected = (n+(1/2))
    E_obtained = (1/(len(local_Es)))*np.sum(local_Es)
    return E_expected, E_obtained

def find_mean_and_std(sigmas_opt, n, h):
    Es = np.zeros((1, 3)) # [expected, obtained, stat_err]

    # Mean energy
    Es[0, 0], Es[0, 1] = average_energies(sigmas_opt, n, h)

    local_enegies = find_local_energies(sigmas_opt, n, h)

    # Statistical Err (chapter 11 notes...)
    sample_varience = (1/(len(local_enegies)-1))*np.sum((local_enegies - Es[0, 1])**2)
    varience_of_mean = sample_varience*(1/(len(local_enegies)))
    Es[0, 2] = np.sqrt(varience_of_mean)
    return Es


'''
# LOAD IN CONTENT:
# FOR LOADING CONTENT BACK INTO ORIGINAL FORM:

import os
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

sigmas_optimised = np.loadtxt(os.path.join(path, "sigmas_optimised.txt"))
red_chi_sqs_by_sig = np.loadtxt(os.path.join(path, "red_chi_sqs_by_sig.txt"))
sigmas = np.loadtxt(os.path.join(path, "sigmas.txt")).reshape(-1, 2, 5)

# Finish the next 2 bullet points...

# Sample 10,000 points...
seeds = [1, 2, -1, -2]

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
'''