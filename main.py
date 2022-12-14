#%%
'''
This is the same code as main3.py. But here, I try to rescale the variables to 
be in order 1.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import tools as tool
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
    return 1 - (np.sin(2*theta)*np.sin(1.267*m*L/(1000*E)))**2

def NNL(theta, m, E_array, count_array, unoscillated_flux_array):
    '''
    Returns NNL.
    '''
    #negative log likelihood using mu_mu_prob (specified above) and poisson distrubition
    n = len(count_array)
    NNL = 0
    for i in range(0,n):
        lambda_i = mu_mu_prob(E_array[i], m, theta)*unoscillated_flux_array[i]
        NNL += lambda_i - count_array[i]*np.log(lambda_i) 
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

def minimize2d(func, x_guess, y_guess, delta=1e-14, max_iterations = 50, a=0.1, b=0.1e-4, starting_direction='x'):
    x_ls_final = [x_guess]
    y_ls_final = [y_guess]
    f_ls_final = [func(x_guess,y_guess)]

    parabola_list_x_final = [np.array([x_guess,x_guess+a,x_guess+2*a])]
    parabola_list_y_final = [np.array([y_guess,x_guess+b,y_guess+2*b])]

    iteration = 0
    while True:
        iteration += 1
        if starting_direction == 'x':
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
        else:
            func_wrt_y = lambda y: func(x_guess, y)
            y_ls, placeholder, y_parabola = minimize1d(func_wrt_y, y_guess,y_guess+b,y_guess+2*b, 1e-15)
            y_guess = y_ls[-1]
            x_ls_final.append(x_guess)
            y_ls_final.append(y_guess)
            f_ls_final.append(func(x_guess, y_guess))
            parabola_list_y_final.append(np.array([y_guess,y_guess+b,y_guess+2*b]))

            func_wrt_x = lambda x: func(x, y_guess)
            x_ls, placeholder, x_parabola = minimize1d(func_wrt_x, x_guess,x_guess+a,x_guess+2*a, 1e-15)
            x_guess = x_ls[-1]
            x_ls_final.append(x_guess)
            y_ls_final.append(y_guess)
            f_ls_final.append(func(x_guess, y_guess))
            parabola_list_x_final.append(np.array([x_guess,x_guess+a,x_guess+2*a]))

        if iteration > 2 and abs(f_ls_final[-1]-f_ls_final[-2]) < delta:
            return np.array(x_ls_final), np.array(y_ls_final), np.array(f_ls_final), np.array(parabola_list_x_final), np.array(parabola_list_y_final)
        elif iteration >= max_iterations:
            return np.array(x_ls_final), np.array(y_ls_final), np.array(f_ls_final), np.array(parabola_list_x_final), np.array(parabola_list_y_final)

def grad_2d(func, parameters, x_guess, y_guess, alpha=1e-5, h=1e-5, delta=1e-14, max_iterations=100):
    '''
    Find minimum using gradient descent.
    '''
    x_ls = [x_guess]
    y_ls = [y_guess]
    f_ls = [func(x_guess,y_guess,*parameters)]

    iteration = 0
    while True:
        xy_cur = np.array([x_ls[-1], y_ls[-1]])
        x = xy_cur[0]
        y = xy_cur[1]

        grad_x = (func(x+h,y,*parameters) - func(x,y,*parameters))/h
        grad_y = (func(x,y+h,*parameters) - func(x,y,*parameters))/h
        grad = np.array([grad_x, grad_y])

        xy_next = xy_cur - alpha*grad

        x_ls.append(xy_next[0])
        y_ls.append(xy_next[1])
        f_ls.append(func(xy_next[0], xy_next[1], *parameters))

        iteration += 1

        if iteration >= max_iterations:
            return np.array(x_ls), np.array(y_ls), np.array(f_ls)
        elif iteration > 2 and abs(f_ls[-1] - f_ls[-2]) <= delta:
            return np.array(x_ls), np.array(y_ls), np.array(f_ls)

def fds_d2f_dxdy(func, x, y, a, b):
    '''
    returns d^2f/dxdy using finite difference scheme with step a in x direction
    and step b in y direction
    '''
    return np.float64((func(x+a,y+b)-func(x+a,y-b)-func(x-a,y+b)+func(x-a,y-b))/(4*a*b))

def fds_d2f_dxdx(func, x, y, a):
    return (func(x+a,y)+func(x-a,y)-2*func(x,y))/(a*a)

def fds_d2f_dydy(func, x, y, b):
    return (func(x,y+b)+func(x,y-b)-2*func(x,y))/(b*b)

def newton2d(func, x_guess, y_guess, a, b, max_iterations=100):

    x_cur = np.array([x_guess,y_guess])
    x_ls_final = [x_guess]
    y_ls_final = [y_guess]

    iteration = 0
    while True:
        iteration += 1

        #Generate and invert the Hessian
        hessian = [[0,0],[0,0]]
        hessian[0][0] = fds_d2f_dxdx(func, x_guess, y_guess, a)
        hessian[1][1] = fds_d2f_dydy(func, x_guess, y_guess, b)
        hessian[0][1] = fds_d2f_dxdy(func, x_guess, y_guess, a, b)
        hessian[1][0] = fds_d2f_dxdy(func, x_guess, y_guess, a, b)
        print(f'H={hessian}')

        inverter = tool.linEq(hessian, [0,0])
        inverter.invert_A()
        inverse_hessian = inverter.return_A_inverse()
        inverse_hessian = np.array(inverse_hessian)
        print(f'H^-1={inverse_hessian}')

        #Generate the gradient vector
        grad_x = (func(x_guess+a,y_guess) - func(x_guess,y_guess))/a
        grad_y = (func(x_guess,y_guess+b) - func(x_guess,y_guess))/b
        grad = np.array([grad_x, grad_y])
        print(f'grad={grad}')

        x_new = x_cur - inverse_hessian@grad

        x_ls_final.append(x_new[0])
        y_ls_final.append(x_new[1])
        x_guess = x_new[0]
        y_guess = x_new[1]

        x_cur = x_new

        if abs(func(x_ls_final[-1],y_ls_final[-1]) - func(x_ls_final[-2],y_ls_final[-2])) <= 1e-10 and iteration > 2:
            return np.array(x_ls_final), np.array(y_ls_final)
        elif iteration >= max_iterations:
            return np.array(x_ls_final), np.array(y_ls_final)

def grad_3d(func, parameters, x_guess, y_guess, z_guess, alpha=1e-5, h=1e-5, delta=1e-14, max_iterations=100):
    '''
    Find minimum using gradient descent.
    '''
    x_ls = [x_guess]
    y_ls = [y_guess]
    z_ls = [z_guess]
    f_ls = [func(x_guess,y_guess,z_guess,*parameters)]

    iteration = 0
    while True:
        xy_cur = np.array([x_ls[-1], y_ls[-1], z_ls[-1]])
        x = xy_cur[0]
        y = xy_cur[1]
        z = xy_cur[2]

        grad_x = (func(x+h,y,z,*parameters) - func(x,y,z,*parameters))/h
        grad_y = (func(x,y+h,z,*parameters) - func(x,y,z,*parameters))/h
        grad_z = (func(x,y,z+h,*parameters) - func(x,y,z,*parameters))/h
        grad = np.array([grad_x, grad_y, grad_z])

        xy_next = xy_cur - alpha*grad

        x_ls.append(xy_next[0])
        y_ls.append(xy_next[1])
        z_ls.append(xy_next[2])
        f_ls.append(func(xy_next[0], xy_next[1], xy_next[2], *parameters))

        iteration += 1

        if iteration >= max_iterations:
            return np.array(x_ls), np.array(y_ls), np.array(z_ls), np.array(f_ls)
        elif iteration > 2 and abs(f_ls[-1] - f_ls[-2]) <= delta:
            return np.array(x_ls), np.array(y_ls), np.array(z_ls), np.array(f_ls)

def fds3d_d2f_dxdy(func, x, y, z, a, b):
    return np.float64((func(x+a,y+b,z)-func(x+a,y-b,z)-func(x-a,y+b,z)+func(x-a,y-b,z))/(4*a*b))

def fds3d_d2f_dxdz(func, x, y, z, a, c):
    return np.float64((func(x+a,y,z+c)-func(x+a,y,z-c)-func(x-a,y,z+c)+func(x-a,y,z-c))/(4*a*c))

def fds3d_d2f_dydz(func, x, y, z, b, c):
    return np.float64((func(x,y+b,z+c)-func(x,y+b,z-c)-func(x,y-b,z+c)+func(x,y-b,z-c))/(4*b*c))

def fds3d_d2f_dxdx(func, x, y, z, a):
    return (func(x+a,y,z)+func(x-a,y,z)-2*func(x,y,z))/(a*a)

def fds3d_d2f_dydy(func, x, y, z, b):
    return (func(x,y+b,z)+func(x,y-b,z)-2*func(x,y,z))/(b*b)

def fds3d_d2f_dzdz(func, x, y, z, c):
    return (func(x,y,z+c)+func(x,y,z-c)-2*func(x,y,z))/(c*c)

def newton3d(func, x_guess, y_guess, z_guess, a, b, c, max_iterations=100):

    x_cur = np.array([x_guess,y_guess,z_guess])
    x_ls_final = [x_guess]
    y_ls_final = [y_guess]
    z_ls_final = [z_guess]

    iteration = 0
    while True:
        iteration += 1

        #Generate and invert the Hessian
        hessian = [[0,0,0],[0,0,0],[0,0,0]]
        hessian[0][0] = fds3d_d2f_dxdx(func, x_guess, y_guess, z_guess, a)
        hessian[1][1] = fds3d_d2f_dydy(func, x_guess, y_guess, z_guess, b)
        hessian[2][2] = fds3d_d2f_dzdz(func, x_guess, y_guess, z_guess, c)
        temp = fds3d_d2f_dxdy(func, x_guess, y_guess, z_guess, a, b)
        hessian[1][0] = temp
        hessian[0][1] = temp
        temp = fds3d_d2f_dxdz(func, x_guess, y_guess, z_guess, a, c)
        hessian[2][0] = temp
        hessian[0][2] = temp
        temp = fds3d_d2f_dydz(func, x_guess, y_guess, z_guess, b, c)
        hessian[2][1] = temp
        hessian[1][2] = temp

        inverter = tool.linEq(hessian, [0,0])
        inverter.invert_A()
        inverse_hessian = inverter.return_A_inverse()
        inverse_hessian = np.array(inverse_hessian)

        #Generate the gradient vector
        grad_x = (func(x_guess+a,y_guess,z_guess) - func(x_guess-a,y_guess,z_guess))/(2*a)
        grad_y = (func(x_guess,y_guess+b,z_guess) - func(x_guess,y_guess-b,z_guess))/(2*b)
        grad_z = (func(x_guess,y_guess,z_guess+c) - func(x_guess,y_guess,z_guess-c))/(2*c)
        grad = np.array([grad_x, grad_y, grad_z])
        print(f'grad={grad}')

        x_new = x_cur - inverse_hessian@grad

        x_ls_final.append(x_new[0])
        y_ls_final.append(x_new[1])
        z_ls_final.append(x_new[2])
        x_guess = x_new[0]
        y_guess = x_new[1]
        z_guess = x_new[2]

        x_cur = x_new

        if iteration > 2 and abs(func(x_ls_final[-1],y_ls_final[-1],z_ls_final[-1]) - func(x_ls_final[-2],y_ls_final[-2],z_ls_final[-2])) <= 1e-10:
            return np.array(x_ls_final), np.array(y_ls_final), np.array(z_ls_final)
        elif iteration >= max_iterations:
            return np.array(x_ls_final), np.array(y_ls_final), np.array(z_ls_final)

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
plt.plot(bin_centers, unoscillated_flux*mu_mu_prob(bin_centers, 2.4, np.pi/4), label='??', c='r')
plt.bar(bin_centers, count, width=0.05, label='Data')
plt.xlabel('Energy (GeV)')
plt.ylabel('?? neutrino count')
plt.grid()
plt.legend()
plt.show()
# %%
'''
Task 3.3: Likelihood function
Create a function for NNL and plot it.
Here, a contour plot is shown.
'''
theta_array = np.linspace(0, np.pi/2, 100)
m_array = np.linspace(1, 4, 100)

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
m = 2.4
NNL_wrt_theta = lambda theta: NNL(theta, m, bin_centers, count, unoscillated_flux)
theta0 = 0.5
theta1 = 0.6
theta2 = 0.7
x_list, y_list, parabola_points= minimize1d(NNL_wrt_theta, theta0, theta1, theta2, 1e-15)

plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='r')
plt.plot(np.linspace(0, np.pi/2, 1000), NNL_wrt_theta(np.linspace(0, np.pi/2, 1000)), '--')
plt.xlim([x_list[-1]-0.1,x_list[-1]+0.1])
plt.ylim([85,100])
plt.xlabel("$??_{23}$")
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

#Now, try coming from the right hand side.
m = 2.4
NNL_wrt_theta = lambda theta: NNL(theta, m, bin_centers, count, unoscillated_flux)
theta0 = 0.8
theta1 = 0.9
theta2 = 1
x_list, y_list, parabola_points= minimize1d(NNL_wrt_theta, theta0, theta1, theta2, 1e-15)
plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='r')
plt.plot(np.linspace(0, np.pi/2, 1000), NNL_wrt_theta(np.linspace(0, np.pi/2, 1000)), '--')
plt.xlim([x_list[-1]-0.1,x_list[-1]+0.1])
plt.ylim([85,100])
plt.xlabel("$??_{23}$")
plt.ylabel(f'NNL($??_{23}$, m={m})')
plt.grid()
plt.show()
#It gives the same result.
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
plt.xlabel("$??_{23}$")
plt.ylabel('NNL')
plt.legend()
plt.show()
# %%
'''
Task 4.1: The univariate method
'''
NNL_wrt_theta_m = lambda theta, m: NNL(theta, m, bin_centers, count, unoscillated_flux)

theta_ls, m_ls, f_ls, theta_parabola, m_parabola = minimize2d(NNL_wrt_theta_m, 0.7, 2.3, b=1e-6)
NNL_wrt_theta = lambda theta: NNL(theta, m_ls[-1], bin_centers, count, unoscillated_flux)
NNL_wrt_m = lambda m: NNL(theta_ls[-1], m, bin_centers, count, unoscillated_flux)
uncertainty_NNL_theta = get_uncertainty(NNL_wrt_theta, theta_ls[-1], 0.05)
uncertainty_NNL_m = get_uncertainty(NNL_wrt_m, m_ls[-1], 0.05)

theta_ls2, m_ls2, f_ls2, theta_parabola2, m_parabola2 = minimize2d(NNL_wrt_theta_m, 0.9, 2.4, b=1e-6)
NNL_wrt_theta = lambda theta: NNL(theta, m_ls2[-1], bin_centers, count, unoscillated_flux)
NNL_wrt_m = lambda m: NNL(theta_ls2[-1], m, bin_centers, count, unoscillated_flux)
uncertainty_NNL_theta2 = get_uncertainty(NNL_wrt_theta, theta_ls2[-1], 0.05)
uncertainty_NNL_m2 = get_uncertainty(NNL_wrt_m, m_ls2[-1], 0.05)

theta_ls_y, m_ls_y, f_ls_y, theta_parabola_y, m_parabola_y = minimize2d(NNL_wrt_theta_m, 0.7, 2.3, starting_direction='y', b=1e-6)
NNL_wrt_theta = lambda theta: NNL(theta, m_ls_y[-1], bin_centers, count, unoscillated_flux)
NNL_wrt_m = lambda m: NNL(theta_ls_y[-1], m, bin_centers, count, unoscillated_flux)
uncertainty_NNL_theta_y = get_uncertainty(NNL_wrt_theta, theta_ls_y[-1], 0.05)
uncertainty_NNL_m_y = get_uncertainty(NNL_wrt_m, m_ls_y[-1], 0.05)

theta_ls2_y, m_ls2_y, f_ls2_y, theta_parabola2_y, m_parabola2_y = minimize2d(NNL_wrt_theta_m, 0.9, 2.4, starting_direction='y', b=1e-6)
NNL_wrt_theta = lambda theta: NNL(theta, m_ls[-1], bin_centers, count, unoscillated_flux)
NNL_wrt_m = lambda m: NNL(theta_ls[-1], m, bin_centers, count, unoscillated_flux)
uncertainty_NNL_theta2_y = get_uncertainty(NNL_wrt_theta, theta_ls2_y[-1], 0.05)
uncertainty_NNL_m2_y = get_uncertainty(NNL_wrt_m, m_ls2_y[-1], 0.05)

theta_uni_plot, m_uni_plot = theta_ls2_y, m_ls2_y

#Plotting the results
theta_array = np.linspace(np.pi/4-0.2,np.pi/4+0.2, 500)
m_array = np.linspace(m_ls[-1]-1.5e-1,m_ls[-1]+1.5e-1, 500)

THETA_ARRAY, M_ARRAY = np.meshgrid(theta_array, m_array)
NNL_contour = NNL(THETA_ARRAY, M_ARRAY, bin_centers, count, unoscillated_flux)
plt.contourf(THETA_ARRAY, M_ARRAY, NNL_contour, 20, cmap='RdGy')
plt.colorbar()
plt.quiver(theta_ls[:-1], m_ls[:-1], theta_ls[1:]-theta_ls[:-1], m_ls[1:]-m_ls[:-1], scale_units='xy', angles='xy', scale=1, color='b')
plt.quiver(theta_ls2[:-1], m_ls2[:-1], theta_ls2[1:]-theta_ls2[:-1], m_ls2[1:]-m_ls2[:-1], scale_units='xy', angles='xy', scale=1, color='g')
plt.xlabel('$??_{23}$')
plt.ylabel('m')
# plt.xlim([np.pi/4-0.1,np.pi/4+0.1])
# plt.ylim([m_ls[-1]-0.5e-1,m_ls[-1]+1e-1])
plt.show()

plt.contourf(THETA_ARRAY, M_ARRAY, NNL_contour, 20, cmap='RdGy')
plt.colorbar()
plt.quiver(theta_ls_y[:-1], m_ls_y[:-1], theta_ls_y[1:]-theta_ls_y[:-1], m_ls_y[1:]-m_ls_y[:-1], scale_units='xy', angles='xy', scale=1, color='b')
plt.quiver(theta_ls2_y[:-1], m_ls2_y[:-1], theta_ls2_y[1:]-theta_ls2_y[:-1], m_ls2_y[1:]-m_ls2_y[:-1], scale_units='xy', angles='xy', scale=1, color='g')
plt.xlabel('$??_{23}$')
plt.ylabel('m')
# plt.xlim([np.pi/4-0.1,np.pi/4+0.1])
# plt.ylim([m_ls[-1]-0.5e-4,m_ls[-1]+1e-4])
plt.show()

# %%
'''
Task 4.2: Simultaneous method

Try using Newton's method (Hessian).
'''
def NNL_proper(theta, m):
    sum = 0
    for i in range(0,len(unoscillated_flux)):
        lamb = unoscillated_flux[i]*mu_mu_prob(bin_centers[i],m,theta)
        sum += lamb - count[i]*np.log(lamb) + np.log(math.factorial(count[i]))
    return sum

theta_ls_newton, m_ls_newton= newton2d(NNL_proper, 0.7, 2.2, 1e-6, 1e-6)
theta_ls_newton2, m_ls_newton2= newton2d(NNL_proper, 0.7, 2.4, 1e-6, 1e-6)
theta_ls_newton3, m_ls_newton3= newton2d(NNL_proper, 0.9, 2.2, 1e-6, 1e-6)
theta_ls_newton4, m_ls_newton4= newton2d(NNL_proper, 0.9, 2.4, 1e-6, 1e-6)

theta_newton_plot, m_newton_plot = theta_ls_newton4, m_ls_newton4

plt.contourf(THETA_ARRAY, M_ARRAY, NNL_contour, 20, cmap='RdGy')
plt.colorbar()
plt.quiver(theta_ls_newton[:-1], m_ls_newton[:-1], theta_ls_newton[1:]-theta_ls_newton[:-1], m_ls_newton[1:]-m_ls_newton[:-1], scale_units='xy', angles='xy', scale=1, color='black')
plt.quiver(theta_ls_newton2[:-1], m_ls_newton2[:-1], theta_ls_newton2[1:]-theta_ls_newton2[:-1], m_ls_newton2[1:]-m_ls_newton2[:-1], scale_units='xy', angles='xy', scale=1, color='green')
plt.quiver(theta_ls_newton3[:-1], m_ls_newton3[:-1], theta_ls_newton3[1:]-theta_ls_newton3[:-1], m_ls_newton3[1:]-m_ls_newton3[:-1], scale_units='xy', angles='xy', scale=1, color='blue')
plt.quiver(theta_ls_newton4[:-1], m_ls_newton4[:-1], theta_ls_newton4[1:]-theta_ls_newton4[:-1], m_ls_newton4[1:]-m_ls_newton4[:-1], scale_units='xy', angles='xy', scale=1, color='orange')
plt.xlabel('$??_{23}$')
plt.ylabel('m')
plt.show()
#%%
'''
Try using gradient descent.
'''
theta_ls_grad, m_ls_grad, p = grad_2d(NNL_proper, [], 0.7, 2.4, 1e-4, 1e-6)
theta_ls_grad2, m_ls_grad2, p = grad_2d(NNL_proper, [], 0.7, 2.2, 1e-4, 1e-6)
theta_ls_grad3, m_ls_grad3, p = grad_2d(NNL_proper, [], 0.9, 2.4, 1e-4, 1e-6)
theta_ls_grad4, m_ls_grad4, p = grad_2d(NNL_proper, [], 0.9, 2.2, 1e-4, 1e-6)

theta_grad_plot, m_grad_plot = theta_ls_grad3, m_ls_grad3

plt.contourf(THETA_ARRAY, M_ARRAY, NNL_contour, 20, cmap='RdGy')
plt.colorbar()
plt.quiver(theta_ls_grad[:-1], m_ls_grad[:-1], theta_ls_grad[1:]-theta_ls_grad[:-1], m_ls_grad[1:]-m_ls_grad[:-1], scale_units='xy', angles='xy', scale=1, color='black')
plt.quiver(theta_ls_grad2[:-1], m_ls_grad2[:-1], theta_ls_grad2[1:]-theta_ls_grad2[:-1], m_ls_grad2[1:]-m_ls_grad2[:-1], scale_units='xy', angles='xy', scale=1, color='green')
plt.quiver(theta_ls_grad3[:-1], m_ls_grad3[:-1], theta_ls_grad3[1:]-theta_ls_grad3[:-1], m_ls_grad3[1:]-m_ls_grad3[:-1], scale_units='xy', angles='xy', scale=1, color='blue')
plt.quiver(theta_ls_grad4[:-1], m_ls_grad4[:-1], theta_ls_grad4[1:]-theta_ls_grad4[:-1], m_ls_grad4[1:]-m_ls_grad4[:-1], scale_units='xy', angles='xy', scale=1, color='orange')
plt.xlabel('$??_{23}$')
plt.ylabel('m')
plt.show()
# %%
'''
Now that we found m and theta that minimises NNL, try plotting
lambda with the new parameters on top of the data histogram.
'''
#ideal data
unoscillated_flux = data['Unoscillated flux']
plt.plot(bin_centers, unoscillated_flux*mu_mu_prob(bin_centers,m_ls_grad3[-1],theta_ls_grad3[-1]), label='??', c='r')
plt.bar(bin_centers, count, width=0.05, label='Data')
plt.xlabel('Energy (GeV)')
plt.ylabel('?? neutrino count')
plt.grid()
plt.legend()
plt.show()

# %%
'''
Task 5: Neutrino Interaction Cross Section

Create a new NNL function that takes into consideration the interaction cross section
when calculating lambda.

Create a 3d gradient descent function.
And create a 3d Hessian minimization function.
'''
def NNL_cross_section(theta, m, a):
    sum = 0
    for i in range(0,len(unoscillated_flux)):
        lamb = unoscillated_flux[i]*mu_mu_prob(bin_centers[i],m,theta)*a*bin_centers[i]
        sum += lamb - count[i]*np.log(lamb) 
    return sum

c = []
def lambda_cross_section(E, unoscillated_flux, theta, m, a):
    lamb_ls = []
    for i in range(0,len(E)):
        lamb = unoscillated_flux[i]*mu_mu_prob(E[i],m,theta)*(a)*E[i]
        lamb_ls.append(lamb)
    return np.array(lamb_ls)

lambda_guess = lambda_cross_section(np.array(bin_centers), np.array(unoscillated_flux), theta_ls_grad4[-1], m_ls_grad4[-1], 1.2413)
plt.plot(np.array(bin_centers), lambda_guess, label='??', c='r')
plt.bar(bin_centers, count, width=0.05, label='Data')
plt.xlabel('Energy (GeV)')
plt.ylabel('?? neutrino count')
plt.grid()
plt.legend()
plt.show()
#%%
###Using gradient descent
theta_list, m_list, a_list, f_list = grad_3d(NNL_cross_section, [], 0.7, 2.4, 2, alpha=1e-4, delta=1e-6, max_iterations=300)
theta_list2, m_list2, a_list2, f_list2 = grad_3d(NNL_cross_section, [], 0.9, 2.4, 2, alpha=1e-4, delta=1e-6, max_iterations=300)
#%%
#Using Hessian
theta_list, m_list, a_list, f_list = newton3d(NNL_cross_section, 0.7, 2.4, 2, a=1e-9, b=1e-9, c=1e-9, max_iterations=100)
theta_list2, m_list2, a_list2, f_list2 = newton3d(NNL_cross_section, 0.9, 2.4, 2, a=1e-9, b=1e-9, c=1e-9, max_iterations=100)
#%%
plt.plot(theta_list, c='red', label='(0.7, 2.4, 2.0)')
plt.plot(theta_list2, c='g', label='(0.9, 2.4, 2.0)')
legend = plt.legend()
legend.set_title('Starting points')
plt.xlabel('Iteration')
plt.ylabel('$??_{23}$')
plt.show()

plt.plot(m_list, c='red', label='(0.7, 2.4, 2.0)')
plt.plot(m_list2, c='g', label='(0.9, 2.4, 2.0)')
legend = plt.legend()
legend.set_title('Starting points')
plt.xlabel('Iteration')
plt.ylabel('m')
plt.show()

plt.plot(a_list, c='red', label='(0.7, 2.4, 2.0)')
plt.plot(a_list2, c='g', label='(0.9, 2.4, 2.0)')
legend = plt.legend()
legend.set_title('Starting points')
plt.xlabel('Iteration')
plt.ylabel('a')
plt.show()

plt.plot(f_list, c='red', label='(0.7, 2.4, 2.0)')
plt.plot(f_list2, c='g', label='(0.7, 2.4, 2.0)')
legend = plt.legend()
legend.set_title('Starting points')
plt.xlabel('Iteration')
plt.ylabel('NNL')
plt.show()

lambda_guess = lambda_cross_section(np.array(bin_centers), np.array(unoscillated_flux), theta_list[-1], m_list[-1], a_list[-1])
plt.plot(np.array(bin_centers), lambda_guess, label='??', c='r')
plt.bar(bin_centers, count, width=0.05, label='Data')
plt.xlabel('Energy (GeV)')
plt.ylabel('?? neutrino count')
plt.grid()
plt.legend()
plt.show()
# %%
'''
Making a final summary plot of all the methods, for 2d
'''

plt.contourf(THETA_ARRAY, M_ARRAY, NNL_contour, 20, cmap='RdGy')
plt.colorbar()
plt.quiver(theta_uni_plot[:-1], m_uni_plot[:-1], theta_uni_plot[1:]-theta_uni_plot[:-1], m_uni_plot[1:]-m_uni_plot[:-1], scale_units='xy', angles='xy', scale=1, color='blue', label='Univariate')
plt.quiver(theta_newton_plot[:-1], m_newton_plot[:-1], theta_newton_plot[1:]-theta_newton_plot[:-1], m_newton_plot[1:]-m_newton_plot[:-1], scale_units='xy', angles='xy', scale=1, color='g', label="Newton's")
plt.quiver(theta_grad_plot[:-1], m_grad_plot[:-1], theta_grad_plot[1:]-theta_grad_plot[:-1], m_grad_plot[1:]-m_grad_plot[:-1], scale_units='xy', angles='xy', scale=1, color='black', label='Grad. des.')
plt.legend()
plt.xlabel('$??_{23}$')
plt.ylabel('m')
plt.show()

# %%
