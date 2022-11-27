#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as math
import tools as tool
# %%
#read in data
data = pd.read_csv('neutrino_energy.csv')

#initialize bin centers
bin_centers = data['Energy lower bound (GeV)']
bin_centers += 0.05/2
count = data['Data to fit']

# %%
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
def NNL(m, theta, E_array, count_array, unoscillated_flux_array):
    #negative log likelihood using mu_mu_prob (specified above) and poisson distrubition
    n = len(count_array)
    NNL = 0
    for i in range(0,n):
        lambda_i = mu_mu_prob(E_array[i], m, theta)*unoscillated_flux_array[i]
        NNL += lambda_i - count_array[i]*np.log(lambda_i) + np.log(math.factorial(count_array[i]))
    return NNL

#trying theta23 between 0 to pi/2 
theta_trials = np.linspace(0,np.pi/2,100)
NNL_trials = []
for theta in theta_trials:
    NNL_trials.append(NNL(2.4e-3, theta, bin_centers, count, unoscillated_flux))

#plotting NNL as a function of theta23
plt.plot(theta_trials, NNL_trials)
plt.xlabel('Theta23')
plt.ylabel('NNL(theta, m=2.4e-3)')
#plot approximate position of the minimum, pi/4:
plt.plot(np.pi/4, NNL(2.4e-3, np.pi/4, bin_centers, count, unoscillated_flux), 'o', c='red')
plt.grid()
plt.show()
#%%
'''
Contour plot of NNL
'''
m_array = np.linspace(2e-3, 3e-3, 100)
theta_array = np.linspace(0, np.pi/2, 100)
M_ARRAY, THETA_ARRAY = np.meshgrid(m_array, theta_array)
NNL_contour = NNL(M_ARRAY, THETA_ARRAY, bin_centers, count, unoscillated_flux)
plt.contourf(THETA_ARRAY, M_ARRAY, NNL_contour, 20, cmap='RdGy')
plt.xlabel('Theta')
plt.ylabel('m')
plt.colorbar()
plt.show()
# %%
'''
Now that we know the minimum of NNL with respect to theta23 is around pi/4,
try and minimize it using parabolic minimizer.
'''
def NNL_theta(theta):
    return NNL(2.4e-3, theta, bin_centers, count, unoscillated_flux)
NNL_obj = tool.functions1d(NNL_theta)
theta_min = NNL_obj.parabolic_min(0.65)

theta_list = np.array(NNL_obj.xmin_ls)
m_list = (2.4e-3)*np.ones(len(theta_list))

m_array = np.linspace(2.2e-3, 2.5e-3, 100)
theta_array = np.linspace(0.55, 0.8, 100)
M_ARRAY, THETA_ARRAY = np.meshgrid(m_array, theta_array)
NNL_contour = NNL(M_ARRAY, THETA_ARRAY, bin_centers, count, unoscillated_flux)
plt.contourf(THETA_ARRAY, M_ARRAY, NNL_contour, 20, cmap='RdGy')
plt.xlabel('Theta')
plt.ylabel('m')
plt.colorbar()
plt.plot(theta_list, m_list, '.')
# plt.quiver(theta_list[:-1], m_list[:-1], theta_list[1:]-theta_list[:-1], m_list[1:]-m_list[:-1], scale_units='xy', angles='xy', scale=1)
plt.show()

# %%
'''
2d minimization using univariate method
'''
def NNL_2d(theta, m):
    return NNL(m, theta, bin_centers, count, unoscillated_flux)

NNL2d_obj = tool.functions2d(NNL_2d, 0.6, 2.5e-3)
theta_min, m_min = NNL2d_obj.parabolic_min2d(max_iterations=15)
# %%
