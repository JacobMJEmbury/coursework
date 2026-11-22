# TRANSFORMTAION METHOD WONT WORK (past n =1) - it is simple to invert for the exponential alone, but any further than that (n=1,2,3,4... cases cant work) and basically impos
# REJECTION SAMPLING: This could work, could define region out to very large x and y, however as we increase n to larger and larger behavior and probability distribution approx will get worse...
# METROPOLIS SAMPLING: This could work better i think, we don't need to worry about the normalising of pdf - this is a benifit as it basically spreads over infinity and we make approx as with the rejection method if we dont intergrate over all bounds...

import numpy as np

n_ddd = 3

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

N_burn = 2000

# Bin the samples
seeds = [1, 2, -1, -2]
samples = []
for seed in seeds:
    sample, _, _ = generate_N_metropolis_points(220000, N_burn=N_burn, sigma=2, n=n_ddd, x0 = 3, I_prev=seed)
    samples.append(sample[::20]) # Take every 20th sample for more non-correlated random samples)

samples = np.concatenate(samples)
print(samples.shape)

min, max = np.argmin(samples), np.argmax(samples)
N_bins = 100
bins = np.zeros((N_bins, 2))

# Define centre point of each bin... Bounds from desmos...
bins[:, 0] = np.linspace(-6, 6, N_bins) # Define bins in range of numbers...

delta_x = np.abs((bins[1, 0] - bins[0, 0]))

# Put number of values into bins...
for val in samples:
    result = np.argmin(np.abs(bins[:, 0] - val))
    bins[result, 1] += 1

# Need to approximately normalise over a large range...
# Choose a wide range to get 99.999% of area...
xs = np.linspace(-50, 50, 1000)
ys = P_x(n=n_ddd, x = xs)

# Intergrate over the whole thing to produce a normalisation for P_x
# HENCE: (produce norm const A)
A = trap_intergration(xs, ys)

# print(A)

xs = np.linspace(bins[0:, 0]-delta_x/2, bins[0:, 0]+delta_x/2, 20)
ys = P_x(n=n_ddd, x = xs)

prob = [trap_intergration(xs[:, i], ys[:, i]) for i in range(N_bins)]

# Now we can do P_x*delta_x = (P_x / A) * delta_x * len(samples) = Expected val
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

red_chi_sq = (1/(mask.sum()-1))*np.sum(((bins[mask, 1] - E[mask])**2)/(E[mask]))

print(red_chi_sq)

# Now I think I will do a red chi sq of the generated N's against the (exp value from) probability distribution...

# Now we have a distribution in bins for len(samples)... 
# Now find the reduced chi-squared values for the expectation value for the prob at each bins centre point...


'''

# Sweep sigma to optimise for the acceptance rate of 44% - I saw this for this acceptance rate: Gelman, Roberts and Gilks, 1996

sigma = np.zeros((50, 2))
sigma[:, 0] = np.linspace(0.5, 5, 50)

for index, sig in enumerate(sigma[:, 0]):
    # Choose n=3 for this - but each optimal sig needs to be found for each n

    total_counts = 0
    accepted_counts = 0

    samples = []
    for seed in seeds:
        sample, total_counts, accepted_counts = generate_N_metropolis_points(220000, N_burn=N_burn, sigma=sig, n=n_ddd, x0 = 3, I_prev=seed, total_counts=total_counts, accepted_counts=accepted_counts)
        samples.append(sample[::20]) # Take every 20th sample for more non-correlated random samples

    sigma[index, 1] = (accepted_counts/(total_counts-N_burn))
    samples = np.concatenate(samples)
    print(samples.shape)

    min, max = np.argmin(samples), np.argmax(samples)
    N_bins = 100
    bins = np.zeros((N_bins, 2))

    # Define centre point of each bin... Bounds from desmos...
    bins[:, 0] = np.linspace(-6, 6, N_bins) # Define bins in range of numbers...

    delta_x = np.abs((bins[1, 0] - bins[0, 0]))

    # Put number of values into bins...
    for val in samples:
        result = np.argmin(np.abs(bins[:, 0] - val))
        bins[result, 1] += 1

    # Need to approximately normalise over a large range...
    # Choose a wide range to get 99.999% of area...
    xs = np.linspace(-50, 50, 1000)
    ys = P_x(n=n_ddd, x = xs)

    # Intergrate over the whole thing to produce a normalisation for P_x
    # HENCE: (produce norm const A)
    A = trap_intergration(xs, ys)

    # print(A)

    xs = np.linspace(bins[0:, 0]-delta_x/2, bins[0:, 0]+delta_x/2, 20)
    ys = P_x(n=n_ddd, x = xs)

    prob = [trap_intergration(xs[:, i], ys[:, i]) for i in range(N_bins)]

    # Now we can do P_x*delta_x = (P_x / A) * delta_x * len(samples) = Expected val
    E = (prob / A) * len(samples)

    #WHY SIGMA_2 = E

    #Think of each sample as a coin flip: it either lands in your bin, with probability p, or it doesn’t. Over N samples, the count follows a binomial distribution, and its variance is N p times one minus p. Your bins are narrow, so p is small, and one minus p is basically one. That leaves N times p, which is exactly your expected count E. So the variance equals E. That’s the Poisson limit.
    
    
    sigma_2 = E

    
    #First, the chi-squared formula assumes each bin count has a roughly bell-shaped spread around E. That holds when E is reasonably big, but with tiny E it’s lopsided. Say E is 0.01 out in the tail and one sample happens to land there: that bin alone adds about 100 to your sum, because you’re dividing by 0.01. A single fluke can swamp everything, so those bins aren’t trustworthy.

    #Second, the opposite case: tail bins where E is basically zero and nothing lands. They add almost nothing to the sum, but each still counts as a degree of freedom. So you’re dividing by a bigger number for no reason, which drags chi-squared down and makes the sampler look better than it is. That’s why you went from 1.38 to 2.33 with the filter.

    #The number 5 is popularised by statistician cochran in 50's


    #print(E>=5) This makes a mask...
    max_count_in_a_bin = bins[[np.argmax(bins[:, 1])], 1].item()
    mask = (E>=5) # cachran threashold
    #print(mask.sum(), N_bins)

    red_chi_sq = (1/(mask.sum()-1))*np.sum(((bins[mask, 1] - E[mask])**2)/(E[mask]))

    print(red_chi_sq)

    # Now I think I will do a red chi sq of the generated N's against the (exp value from) probability distribution...

    # Now we have a distribution in bins for len(samples)... 
    # Now find the reduced chi-squared values for the expectation value for the prob at each bins centre point...


import matplotlib.pyplot as plt
plt.plot(sigma[:, 0], sigma[:, 1])
plt.show()

optimal_sigma = sigma[np.argmin(np.abs(sigma[:, 1] - 0.44)), 0]
print(optimal_sigma)

'''

optimal_sig = 2.1530612244897958

