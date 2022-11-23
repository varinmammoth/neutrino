#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %%
data = pd.read_csv('neutrino_energy.csv')

bin_centers = data['Energy lower bound (GeV)']
bin_centers += 0.05/2
count = data['Data to fit']
plt.bar(bin_centers, count, width=0.1)
plt.plot(4.6-0.05/2, 14, 'o', c='red')
plt.show()
# %%
def prob(E, m=2.4e-3, theta=np.pi/4, L=295):
    return 1 - (np.sin(2*theta)*np.sin(1.267*m*L/E))**2

#ideal data
unoscillated_flux = data['Unoscillated flux']
plt.plot(bin_centers, unoscillated_flux*prob(bin_centers), label='Ideal', c='g')
plt.bar(bin_centers, count, width=0.05, label='Actual data')
plt.xlabel('Energy (GeV)')
plt.ylabel('mu neutrino count')
plt.grid()
plt.legend()
plt.show()
# %%
