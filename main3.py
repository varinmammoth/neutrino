#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
# %%
def mu_mu_prob(E, m=2.4e-3, theta=np.pi/4, L=295):
    """Probability of neutrino not oscillating.

    Args:
        E (float): Energy
        m (float, optional): Mas. Defaults to 2.4e-3.
        theta (float, optional): theta. Defaults to np.pi/4.
        L (int, optional): L. Defaults to 295.

    Returns:
        float: Probability of neutrino not oscillating.
    """
    #probability of mu neutrino still being a mu neutrino
    return 1 - (np.sin(2*theta)*np.sin(1.267*m*L/E))**2

def NNL(theta, m, E_array, count_array, unoscillated_flux_array):
    '''
    Returns NNL.
    '''
    #negative log likelihood using mu_mu_prob (specified above) and poisson distrubition
    n = len(count_array)
    NNL = 0
    for i in range(0,n):
        lambda_i = mu_mu_prob(E_array[i], m, theta)*unoscillated_flux_array[i]
        NNL += lambda_i - count_array[i]*np.log(lambda_i) + np.log(math.factorial(count_array[i]))
        # NNL += lambda_i - count_array[i]*np.log(lambda_i) 
    return NNL

def lagrange_poly(x, x_data, y_data, n):
    '''
    function of n-th degree Lagrange polynomial.
    '''
    sum = 0
    for i in range(0,n+1):
        product = 1
        for j in range(0,n+1):
            if i != j:
                product *= (x-x_data[j])/(x_data[i]-x_data[j])
        sum += product*y_data[i]  
    return sum

def secant(func, x1, x2):
    '''
    Finding zeros of func using the secant method.
    x1 and x2 are initial guesses.
    '''
    while True:
        f1 = func(x1)
        f2 = func(x2)
        temp = x1
        x1 = x1-(x1-x2)*f1/(f1-f2)
        x2 = temp
        if abs(f1) < 1e-6:
            return x1

def get_uncertainty(func, xmin, uncertainty_guess):
    '''
    Calculate uncertainty by determining what x values causes the
    function to change by 0.5
    '''
    uncer_func = lambda x: func(x) - (func(xmin)+0.5)
    uncer_plus = secant(uncer_func, xmin+uncertainty_guess, xmin+1.1*uncertainty_guess)
    uncer_minus = secant(uncer_func, xmin-uncertainty_guess, xmin-1.1*uncertainty_guess)
    uncer = [uncer_plus-xmin, xmin-uncer_minus]

    return uncer

def get_uncertainty_parabola(x_min, y_min, x_parabola_points, y_parabola_points):
    parabola = lambda x: lagrange_poly(x, x_parabola_points, y_parabola_points, 2)
    h = 1e-4
    second_derivative = (parabola(x_min+h) + parabola(x_min-h) - 2*parabola(x_min))/(h**2)
    return np.sqrt(1/second_derivative)

def minimize1d(func, x0, x1, x2, delta, max_iterations=100):
    '''
    Minimize in 1d using parabolic method.
    '''
    iteration = 0

    points = np.array([x0,x1,x2])
    f_points = func(points)

    x_list = [points[np.argmin(f_points)]]
    y_list = [np.min(f_points)]
    three_points = [np.array([x0,x1,x2])]

    while True:
        x0 = points[0]
        x1 = points[1]
        x2 = points[2]
        y0 = f_points[0]
        y1 = f_points[1]
        y2 = f_points[2]
        x_new = (1/2)*((x2**2-x1**2)*y0 + (x0**2-x2**2)*y1 + (x1**2-x0**2)*y2)/((x2-x1)*y0 + (x0-x2)*y1 + (x1-x0)*y2)

        points = np.append(points, x_new)
        f_points = np.append(f_points, func(x_new))
        max_index = np.argmax(f_points)
        points = np.delete(points, max_index)
        f_points = np.delete(f_points, max_index)

        x_list.append(points[np.argmin(f_points)])
        y_list.append(np.min(f_points))
        three_points.append(points)

        iteration += 1
        
        if iteration > 2 and abs(y_list[-1] - y_list[-2]) <= delta:
            return np.array(x_list), np.array(y_list), np.array(three_points)
        elif iteration >= max_iterations:
            return np.array(x_list), np.array(y_list), np.array(three_points)

def minimize2d(func, x_guess, y_guess, delta=1e-14, max_iterations = 50, a=0.1, b=0.1e-4):
    x_ls_final = [x_guess]
    y_ls_final = [y_guess]
    f_ls_final = [func(x_guess,y_guess)]

    parabola_list_x_final = [np.array([x_guess,x_guess+a,x_guess+2*a])]
    parabola_list_y_final = [np.array([y_guess,x_guess+b,y_guess+2*b])]

    iteration = 0
    while True:
        iteration += 1

        func_wrt_x = lambda x: func(x, y_guess)
        x_ls, placeholder, x_parabola = minimize1d(func_wrt_x, x_guess,x_guess+a,x_guess+2*a, 1e-15)
        x_guess = x_ls[-1]
        x_ls_final.append(x_guess)
        y_ls_final.append(y_guess)
        f_ls_final.append(func(x_guess, y_guess))
        parabola_list_x_final.append(np.array([x_guess,x_guess+a,x_guess+2*a]))

        func_wrt_y = lambda y: func(x_guess, y)
        y_ls, placeholder, y_parabola = minimize1d(func_wrt_y, y_guess,y_guess+b,y_guess+2*b, 1e-15)
        y_guess = y_ls[-1]
        x_ls_final.append(x_guess)
        y_ls_final.append(y_guess)
        f_ls_final.append(func(x_guess, y_guess))
        parabola_list_y_final.append(np.array([y_guess,y_guess+b,y_guess+2*b]))

        if iteration > 2 and abs(f_ls_final[-1]-f_ls_final[-2]) < delta:
            return np.array(x_ls_final), np.array(y_ls_final), np.array(f_ls_final), np.array(parabola_list_x_final), np.array(parabola_list_y_final)
        elif iteration >= max_iterations:
            return np.array(x_ls_final), np.array(y_ls_final), np.array(f_ls_final), np.array(parabola_list_x_final), np.array(parabola_list_y_final)
# %%
'''
Task 3.1: Data
and 
Task 3.2: Fit function
Read in the data and create a histogram.
Plotting lambda over the data.
'''
'''
Read data and shift bins to center.
'''
#read in data
data = pd.read_csv('neutrino_energy.csv')

#initialize bin centers
bin_centers = data['Energy lower bound (GeV)']
bin_centers += 0.05/2
count = data['Data to fit']

#ideal data
unoscillated_flux = data['Unoscillated flux']
plt.plot(bin_centers, unoscillated_flux*mu_mu_prob(bin_centers), label='λ', c='r')
plt.bar(bin_centers, count, width=0.05, label='Data')
plt.xlabel('Energy (GeV)')
plt.ylabel('µ neutrino count')
plt.grid()
plt.legend()
plt.show()

#The probability of no oscillation as a function of energy
energies = np.linspace(0.025,10,100000)
plt.subplot(2,1,1)
plt.plot(energies, mu_mu_prob(energies), c='red')
plt.ylabel('P(μ→μ)')
plt.xlim([0.025,10])
plt.grid()

plt.subplot(2,1,2)
plt.plot(energies, mu_mu_prob(energies), c='red')
plt.xlim([0.025,10])
plt.xlabel('Energy (GeV)')
plt.ylabel('P(μ→μ)')
plt.xscale('log')
plt.grid()
plt.show()
# %%
'''
Task 3.3: Likelihood function
Create a function for NNL and plot it.
Here, a contour plot is shown.
'''
theta_array = np.linspace(0, np.pi/2, 100)
m_array = np.linspace(1e-3, 4e-3, 100)

THETA_ARRAY, M_ARRAY = np.meshgrid(theta_array, m_array)
NNL_contour = NNL(THETA_ARRAY, M_ARRAY, bin_centers, count, unoscillated_flux)
plt.contourf(THETA_ARRAY, M_ARRAY, NNL_contour, 20, cmap='RdGy')
plt.xlabel('Theta')
plt.ylabel('m')
plt.colorbar()
plt.show()
# %%
'''
Task 3.4: Minimise
First, we fix m at 0.0024.
Then we try to minimise NNL with respect to theta using parabolic method.
'''
#Make a new NNL function that takes in only theta as the argument.
m = 0.0024
NNL_wrt_theta = lambda theta: NNL(theta, m, bin_centers, count, unoscillated_flux)
theta0 = 0.5
theta1 = 0.6
theta2 = 0.7
x_list, y_list, parabola_points= minimize1d(NNL_wrt_theta, theta0, theta1, theta2, 1e-15)

plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='r')
plt.plot(np.linspace(0, np.pi/2, 1000), NNL_wrt_theta(np.linspace(0, np.pi/2, 1000)), '--')
plt.xlim([x_list[-1]-0.1,x_list[-1]+0.1])
plt.ylim([607.5, 620])
plt.xlabel("$θ_{23}$")
plt.ylabel(f'NNL(theta, m={m})')
plt.grid()
plt.show()

#An animation showing what is happening:
# theta = np.linspace(x_list[-1]-0.1,x_list[-1]+0.1,150)
# for i, points in enumerate(parabola_points):
#     plt.plot(theta, lagrange_poly(theta, points, NNL_wrt_theta(points), 2))
#     plt.plot(x_list[i], y_list[i], 'o', c='red')
#     plt.plot(theta, NNL_wrt_theta(theta))
#     plt.ylim([608, 620])
#     plt.pause(2)
#     plt.clf()
# %%
'''
Task 3.5: Finding the accuracy of fit result

Method 1: Finding at which values of theta causes NNL_min to become NNL_min+-0.5.

Method 2: Using a Taylor expansion f(x+d) up to the second-derivative term.
In our case, f-->last parabola generated from parabolic method, x-->theta. 
We equate f(x+d)=0.5, solve for d. d is the uncertainty.
'''
#Method 1:
uncertainty_NNL = get_uncertainty(NNL_wrt_theta, x_list[-1], 0.05)
print(f'The minimum theta is {x_list[-1]} with uncertainty +- {uncertainty_NNL[0]} using NNL method.')

#Method 2:
uncertainty_parabola = get_uncertainty_parabola(x_list[-1], y_list[-1], parabola_points[-1], NNL_wrt_theta(np.array(parabola_points[-1])))
print(f'The minimum theta is {x_list[-1]} with uncertainty +- {uncertainty_parabola} using parabola method')

'''
We see that the uncertainty using method 2 is slightly larger than when using method 1.
The reason is because near the minimum, NNL is very flat, so the parabola used to approximate
this region will be very wide. So, to go 1 s.d. away on this parabola would need a bigger shift
to the left and right of the minimum.

The graph below shows the actual NNL function vs the parabola used to approximate it at the minimum.
'''
theta = np.linspace(x_list[-1]-0.1,x_list[-1]+0.1,150)
plt.plot(theta, NNL_wrt_theta(theta), label='NNL')
plt.plot(theta, lagrange_poly(theta, parabola_points[-1], NNL_wrt_theta(np.array(parabola_points[-1])), 2), label='Parabola')
plt.grid()
plt.xlabel("$θ_{23}$")
plt.ylabel('NNL')
plt.legend()
plt.show()
# %%
'''
Task 4.1: The univariate method
'''
NNL_wrt_theta_m = lambda theta, m: NNL(theta, m, bin_centers, count, unoscillated_flux)

theta_ls, m_ls, f_ls, theta_parabola, m_parabola = minimize2d(NNL_wrt_theta_m, 0.5, 2.4e-3)
theta_ls2, m_ls2, f_ls2, theta_parabola2, m_parabola2 = minimize2d(NNL_wrt_theta_m, 0.5, 2.31e-3)

#Plotting the results
theta_array = np.linspace(np.pi/4-0.1,np.pi/4+0.1, 500)
m_array = np.linspace(m_ls[-1]-0.5e-4,m_ls[-1]+1e-4, 500)

THETA_ARRAY, M_ARRAY = np.meshgrid(theta_array, m_array)
NNL_contour = NNL(THETA_ARRAY, M_ARRAY, bin_centers, count, unoscillated_flux)
plt.contourf(THETA_ARRAY, M_ARRAY, NNL_contour, 20, cmap='RdGy')
plt.colorbar()
plt.quiver(theta_ls[:-1], m_ls[:-1], theta_ls[1:]-theta_ls[:-1], m_ls[1:]-m_ls[:-1], scale_units='xy', angles='xy', scale=1, color='b')
plt.quiver(theta_ls2[:-1], m_ls2[:-1], theta_ls2[1:]-theta_ls2[:-1], m_ls2[1:]-m_ls2[:-1], scale_units='xy', angles='xy', scale=1, color='g')
plt.xlabel('Theta')
plt.ylabel('m')
plt.xlim([np.pi/4-0.1,np.pi/4+0.1])
plt.ylim([m_ls[-1]-0.5e-4,m_ls[-1]+1e-4])
plt.show()
# %%
