#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as math
import tools as tool
# %%
'''
Read data and shift bins to center.
'''
#read in data
data = pd.read_csv('neutrino_energy.csv')

#initialize bin centers
bin_centers = data['Energy lower bound (GeV)']
bin_centers += 0.05/2
count = data['Data to fit']

# %%
'''
Define probability function.
Finding lambda as a function of energy, for a fixed m and theta.
Plot this lambda over the raw data.
'''
def mu_mu_prob(E, m=2.4e-3, theta=np.pi/4, L=295):
    #probability of mu neutrino still being a mu neutrino
    return 1 - (np.sin(2*theta)*np.sin(1.267*m*L/E))**2

#ideal data
unoscillated_flux = data['Unoscillated flux']
plt.plot(bin_centers, unoscillated_flux*mu_mu_prob(bin_centers), label='Ideal', c='r')
plt.bar(bin_centers, count, width=0.05, label='Actual data')
plt.xlabel('Energy (GeV)')
plt.ylabel('mu neutrino count')
plt.grid()
plt.legend()
plt.show()
# %%
'''
Define NNL function.
Keep m fixed and plot it as a function of theta.
'''
def NNL(theta, m, E_array, count_array, unoscillated_flux_array):
    #negative log likelihood using mu_mu_prob (specified above) and poisson distrubition
    n = len(count_array)
    NNL = 0
    for i in range(0,n):
        lambda_i = mu_mu_prob(E_array[i], m, theta)*unoscillated_flux_array[i]
        NNL += lambda_i - count_array[i]*np.log(lambda_i) + np.log(math.factorial(count_array[i]))
        # NNL += lambda_i - count_array[i]*np.log(lambda_i) 
    return NNL

m = 2.4e-3  #fixing m
#plot for theta=0 to pi/2
theta_trials = np.linspace(0,np.pi/2,100)
NNL_trials = []
for theta in theta_trials:
    NNL_trials.append(NNL(theta, 2.4e-3, bin_centers, count, unoscillated_flux))
plt.plot(theta_trials, NNL_trials)
plt.xlabel('Theta')
plt.ylabel(f'NNL(theta, m={m})')
plt.grid()
plt.show()
# %%
'''
Out of interest, plot a contour of NNL as a function of theta and m.
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
Make 1D minimizer using the parabolic method.
'''
def minimize1d(func, x0, x1, x2, delta, max_iterations=100):
    iteration = 0

    points = np.array([x0,x1,x2])
    f_points = func(points)

    x_list = [points[np.argmin(f_points)]]
    y_list = [np.min(f_points)]

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

        iteration += 1

        if iteration > 2 and abs(y_list[-1] - y_list[-2]) <= delta:
            return np.array(x_list), np.array(y_list)
        elif iteration >= max_iterations:
            return np.array(x_list), np.array(y_list)
#%%
'''
Testing the 1d parabolic minimizer
'''
x = np.linspace(4.5, 8.5, 100)
y = lambda x: (x-5)*(x-6)*(x-7)*(x-8)

x0 = 4
x1 = 4.1
x2 = 4.2
x_list, y_list = minimize1d(y, x0, x1, x2, 1e-15)
plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='r')

x0 = 10
x1 = 10.1
x2 = 10.2
x_list, y_list = minimize1d(y, x0, x1, x2, 1e-15)
plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='r')

plt.plot(x, y(x), '--', c='grey')
plt.ylim([-2,2])
plt.grid()
plt.show()

# %%
'''
Applying the 1D parabolic minimizer wrt theta on NNL,
keeping m fixed.
'''
#Make a new NNL function that takes in only theta as the argument.
# m = 2.4e-3
m = 0.0024
NNL_wrt_theta = lambda theta: NNL(theta, m, bin_centers, count, unoscillated_flux)
theta0 = 0.5
theta1 = 0.6
theta2 = 0.7
x_list, y_list = minimize1d(NNL_wrt_theta, theta0, theta1, theta2, 1e-15)

plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='r')
plt.plot(np.linspace(0, np.pi/2, 1000), NNL_wrt_theta(np.linspace(0, np.pi/2, 1000)), '--')
plt.xlim([x_list[-1]-0.1,x_list[-1]+0.1])
plt.ylim([607.5, 620])
plt.xlabel('Theta')
plt.ylabel(f'NNL(theta, m={m})')
plt.grid()
plt.show()
# %%
'''
Estimate the uncertainy in theta_min

!!!!!
Skip for now
!!!!!
'''
# %%
'''
Make 2d minimizer using the parabolic method.
Taking the first step in the theta direction.
'''
#Going in theta direction first
def minimize2d_NNL(theta_guess, m_guess, delta=1e-14, max_iterations = 50):
    theta_list_final = [theta_guess]
    m_list_final = [m_guess]
    NNL_list_final = [NNL(theta_guess, m_guess, bin_centers, count, unoscillated_flux)]

    parabola_list_final = []
    
    iteration = 0
    while True:
        iteration += 1

        NNL_wrt_theta = lambda theta: NNL(theta, m_guess, bin_centers, count, unoscillated_flux)
        theta_list, placeholder = minimize1d(NNL_wrt_theta, theta_guess, theta_guess+0.1, theta_guess+0.2, 1e-15)
        theta_guess = theta_list[-1]
        theta_list_final.append(theta_guess)
        m_list_final.append(m_guess)
        NNL_list_final.append(NNL(theta_guess, m_guess, bin_centers, count, unoscillated_flux))
        
        NNL_wrt_m = lambda m: NNL(theta_guess, m, bin_centers, count, unoscillated_flux)
        m_list, placeholder = minimize1d(NNL_wrt_m, m_guess, m_guess+0.1e-4, m_guess+0.2e-4, 1e-15)
        m_guess = m_list[-1]
        theta_list_final.append(theta_guess)
        m_list_final.append(m_guess)
        NNL_list_final.append(NNL(theta_guess, m_guess, bin_centers, count, unoscillated_flux))

        if iteration > 2 and abs(NNL_list_final[-1]-NNL_list_final[-2]) < delta:
            return np.array(theta_list_final), np.array(m_list_final), np.array(NNL_list_final)
        elif iteration >= max_iterations:
            return np.array(theta_list_final), np.array(m_list_final), np.array(NNL_list_final)
            
#Uncomment to try different initial guesses
#1           
# theta_list, m_list, NNL_list = minimize2d_NNL(0.5, 2.4e-3)
# theta_list2, m_list2, NNL_list2 = minimize2d_NNL(0.9, 2e-3)
#2
theta_list, m_list, NNL_list = minimize2d_NNL(0.5, 2.4e-3)
theta_list2, m_list2, NNL_list2 = minimize2d_NNL(0.5, 2.3e-3)

#Plotting the results
theta_array = np.linspace(np.pi/4-0.1,np.pi/4+0.1, 500)
m_array = np.linspace(m_list[-1]-0.5e-4,m_list[-1]+1e-4, 500)

THETA_ARRAY, M_ARRAY = np.meshgrid(theta_array, m_array)
NNL_contour = NNL(THETA_ARRAY, M_ARRAY, bin_centers, count, unoscillated_flux)
plt.contourf(THETA_ARRAY, M_ARRAY, NNL_contour, 20, cmap='RdGy')
plt.colorbar()
plt.quiver(theta_list[:-1], m_list[:-1], theta_list[1:]-theta_list[:-1], m_list[1:]-m_list[:-1], scale_units='xy', angles='xy', scale=1, color='b')
plt.quiver(theta_list2[:-1], m_list2[:-1], theta_list2[1:]-theta_list2[:-1], m_list2[1:]-m_list2[:-1], scale_units='xy', angles='xy', scale=1, color='g')
plt.xlabel('Theta')
plt.ylabel('m')
plt.xlim([np.pi/4-0.1,np.pi/4+0.1])
plt.ylim([m_list[-1]-0.5e-4,m_list[-1]+1e-4])
plt.show()

print(f'The minima (theta, m) are at:') 
print(f'({theta_list[-1]}, {m_list[-1]}) with NNL={NNL_list[-1]}')
print(f'({theta_list2[-1]}, {m_list2[-1]}) with NNL={NNL_list2[-1]}')

#Plot NNL as function of theta for fixed m=0.0023
m_min = m_list[-1]
plt.plot(np.linspace(0, np.pi/2, 1000), NNL(np.linspace(0, np.pi/2, 1000), m_min, bin_centers, count, unoscillated_flux),  '--')
plt.plot(theta_list[-1], NNL(theta_list[-1], m_min, bin_centers, count, unoscillated_flux), 'o')
plt.plot(theta_list2[-1], NNL(theta_list2[-1], m_min, bin_centers, count, unoscillated_flux), 'o')
plt.xlabel('Theta')
plt.ylabel(f'NNL(m={m_list2[-1]}')
plt.xlim([np.pi/4-0.1, np.pi/4+0.1])
plt.ylim([602.9,603.25])
plt.grid()
plt.show()
# %%
'''
Make a 2d gradient descent function
'''
def grad_2d(func, parameters, x_guess, y_guess, alpha=1e-5, h=1e-5, delta=1e-14, max_iterations=100):
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
        elif np.linalg.norm(xy_cur-xy_next) <= delta:
            return np.array(x_ls), np.array(y_ls), np.array(f_ls)

'''
Test the function
'''
def test_function(x,y,a,b,c):
    return a*(np.sin(x)**2) + b*(np.cos(y)**2) + c

x_ls_grad, y_ls_grad, f_ls_grad = grad_2d(test_function, [2,5,1], 5, 6, alpha=1e-3, h=1e-5, delta=1e-14, max_iterations=5000)

#Plotting the results
x_array = np.linspace(4,8, 100)
y_array = np.linspace(4,8, 100) 

X_ARRAY, Y_ARRAY = np.meshgrid(x_array, y_array)
F_COUNTOUR = test_function(X_ARRAY, Y_ARRAY, *[2,5,1])
plt.contourf(X_ARRAY, Y_ARRAY, F_COUNTOUR, 20, cmap='RdGy')
plt.colorbar()
plt.quiver(x_ls_grad[:-1], y_ls_grad[:-1], x_ls_grad[1:]-x_ls_grad[:-1], y_ls_grad[1:]-y_ls_grad[:-1], scale_units='xy', angles='xy', scale=1, color='b')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# %%
'''
Apply the 2D gradient descent on NNL.
'''
#Minimising
params = [bin_centers, count, unoscillated_flux]
theta_ls_grad, m_ls_grad, NNL_ls_grad = grad_2d(NNL, params, 0.775, 2.32e-3, alpha=1e-5, h=1e-5, delta=1e-14, max_iterations=100)

#Plotting the results
#Plotting the results
theta_array = np.linspace(np.pi/4-0.1,np.pi/4+0.1, 500)
m_array = np.linspace(m_list[-1]-0.5e-4,m_list[-1]+1e-4, 500)

THETA_ARRAY, M_ARRAY = np.meshgrid(theta_array, m_array)
NNL_contour = NNL(THETA_ARRAY, M_ARRAY, bin_centers, count, unoscillated_flux)
plt.contourf(THETA_ARRAY, M_ARRAY, NNL_contour, 20, cmap='RdGy')
plt.colorbar()
# plt.quiver(theta_ls_grad[:-1], m_ls_grad[:-1], theta_ls_grad[1:]-theta_ls_grad[:-1], m_ls_grad[1:]-m_ls_grad[:-1], scale_units='xy', angles='xy', scale=1, color='b')
plt.plot(theta_ls_grad, m_ls_grad)
plt.xlabel('Theta')
plt.ylabel('m')
plt.xlim([np.pi/4-0.1,np.pi/4+0.1])
plt.ylim([m_list[-1]-0.5e-4,m_list[-1]+1e-4])
plt.show()
# %%
'''
Try Newton Method
'''
def annealing2d(func, params, x_guess, y_guess):
    T_ls = np.arange(0, 300, 1)
    T_ls = np.flip(T_ls)