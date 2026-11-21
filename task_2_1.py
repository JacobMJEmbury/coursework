# Choose a eigenfunction to test with:
# Choose n = 0 for ease...
# Hence psi = (1)exp[-x**2/2]

import numpy as np
import matplotlib.pyplot as plt
# choose x in range [-4, 4] as on desmos we get most of distribution that has non-negligable gradient in this region 
# Define float64 so we have high resolution for checking the small difference errors in our approx. methods

x = np.float64(np.linspace(-4, 4, 1000))

def psi(x):
    return np.float64(np.exp(-(x**2)/2))

# Define LHS of equation
n = 0 # Define n = 0 (as mentioned above)
energy = np.float64(n + 1/2) # Energy should be equal to this at every point x...

# Define three difference methods - this took a long ass time to do - INCLUDE IN REPORT THE EQUATIONS?
# ASK: Should i include equations for difference method i chose?

def forward_difference(x, h):
    y_double_d = np.float64(np.zeros(len(x)))
    y_2h = psi(x+2*h)
    y_h = psi(x+h)
    y = psi(x)

    y_double_d = (y_2h - 2*y_h + y)/h**2

    return y_double_d


def central_difference_3_points(x, h):
    y_double_d = np.float64(np.zeros(len(x)))
    y_plus_h = psi(x+h)
    y_minus_h = psi(x-h)
    y = psi(x)

    y_double_d = (y_plus_h - 2*y + y_minus_h)/h**2

    return y_double_d


def central_difference_5_points(x, h):
    y_double_d = np.float64(np.zeros(len(x)))
    y_plus_2h = psi(x+2*h)
    y_plus_h = psi(x+h)
    y_minus_h = psi(x-h)
    y_minus_2h = psi(x-2*h)
    y = psi(x)

    y_double_d = (-(15/6)*y + (4/3)*(y_plus_h+y_minus_h) - (1/12)*(y_plus_2h+y_minus_2h))/h**2

    return y_double_d


# Compare the difference methods to the true value of energy and average over all the points in x range...

def check_error(x, energy, method):
    E_calc = (1/psi(x))*(-1/2)*(method) + (1/2)*(x**2)
    E_error = np.abs(E_calc - energy)

    average_error = np.sum(E_error) / len(E_error)
    return average_error

methods = [forward_difference, central_difference_3_points, central_difference_5_points]
#hs = np.linspace(1, 10**(-15), 10000) # This is so stupid - wont yeild good results for what trying to do...
hs = np.logspace(-15, 0, 1000) # FIX: this spreads points evenly in log h
errors = np.zeros((len(hs), 3))

# Iterate through all the methods and print out the h for the lowest error... (along with the error itself...)
for index, method in enumerate(methods):
    errors[:, index] = [check_error(x, energy, method(x, h)) for h in hs]


    min_index = np.argmin(errors[:, index])
    print("METHOD: ", method, "h size:", hs[min_index], "error:", errors[min_index, index])

    #plt.plot(error_by_h[:, 0], error_by_h[:, 1])
    '''    
    plt.loglog(hs, errors)
    plt.show()
    plt.plot((hs), np.log10(errors))
    min_index = np.argmin(errors)
    print("METHOD: ", method, "h size:", hs[min_index], "error:", errors[min_index])
    plt.show()
    '''

# Plot the 3 curves on one axis and describ them...
max = len(errors[:, 0])#-100
plt.loglog(hs[:max], errors[:max, 0], label="Forward difference")
plt.loglog(hs, errors[:, 1],  label="Central difference (3 points)")
plt.loglog(hs, errors[:, 2], label="Central difference (5 points)")
plt.xlabel("Log(h)")
plt.ylabel("Log(errors)")
plt.show()


# Things of note from these graphs:
# -> The derivations for the three solutions give the following big O errs...
# |--> Forward difference: O(h) = h*f''
# |--> Central Difference (3 points) = O(h**2) = (h**2/12) * f''''
# |--> Central Difference (5 points) = O(h**4) = (-1/90)*(h**4)* f''''''

# -> This means as the h decreases, the log_10(h) should scale linearly with h with a gradient of the indicie of each of em'

''' FROM PRINT STATEMENT ABOVE:
METHOD:  <function forward_difference at 0x10b20f600> h size: 3.93182875570577e-06 error: 1.3329416041639674e-05
METHOD:  <function central_difference_3_points at 0x10bdf47c0> h size: 0.00010495932305582266 error: 1.6925891691166316e-08
METHOD:  <function central_difference_5_points at 0x10bdf4860> h size: 0.0026146732118010867 error: 3.928972974742706e-11
'''

# To obtain the minimum error - just took the argmin of the error (this is the turning point on the graph...)
# It is pretty clear that the central difference with 5 points yeilds not only the best error (smallest - by 3 orders) but also the highest h to get there (which is more optimal - EXPLAIN IN LAB SCRIPT...)

# Why do we get like an minimum point - it goes down then goes up...
# -> As we get down to the minimum point we find that because we are using float 64 (which is double precision)
# -> we get to a point where the subtraction between the psi(x) and psi(x+/-h) and psi(x+/- 2h) gets so small that we reach the 10**-16 double precision limit... (negligable difference - computer cant differentiate the numerator well any further)
# -> This error in the double derivitive is then divided by h**2 (a very small number now) - causing the scaling of the error on the LHS to be apporx. -2 (O(h**-2))
# -> I think the small round off value we get has a very large error which is then excentuated by the fact that we divide by h**2 (which is an even smaller number) - blowing up the h approx... That is probs why all 3 lines have the same approx. 2 gradient line as h keeps decreasing...

#### TO COMPARE THE DIFFERENT METHODS ERROR ORDER:
# Now we should obtain the gradient of the linearly decreasing dection from h=1->h=optimal h size to show big O derivations are true...

def determine_gradient(h, errs):
    # We define here a min and max (very rough) - this just clips from the largest h to the min-h value... We get some edge effects (not sure what from - LOOK INTO THIS)... This just means we get a strait line to resolve the big O efficiency... 
    min = (np.argmin(errs)) + 50
    max = len(errs)-50
    #print(min, max)
    # Very rudementary - but doing it manually and quickly - not actual result ill probs mention...
    errs = np.log10(errs[min:max])
    h = np.log10(h[min:max])
    return (errs[-1] - errs[0]) / (h[-1] - h[0])

for i in range(0,3):
    print("The", methods[i], "has error O(h**" + str(determine_gradient(hs, errors[:, i])) + ")")

max = (np.argmin(errors[:, 2])) - 50
errs = np.log10(errors[0:max, 2])
h = np.log10(hs[0:max])
grad =  (errs[-1] - errs[0]) / (h[-1] - h[0])
print("The gradient of the LHS slope (approx equal for all methods):", grad, "hence we get O(h**" + str(grad) + ")")

# Hence I will use the 5 point central difference is the one to use...
# Slopes show is 4th order... h size: 0.0026146732118010867 error: 3.928972974742706e-11
