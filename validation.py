#%%
'''
This is a file to test the functions created.
'''
import numpy as np
import matplotlib.pyplot as plt
import main3 as m
# %%
'''
Test of 1d parabolic minimiser.
'''
plt.subplot(2,2,1)
x = np.linspace(4.5, 8.5, 100)
y = lambda x: (x-5)*(x-6)*(x-7)*(x-8)
x0 = 4
x1 = 4.1
x2 = 4.2
x_list, y_list, parabola_points = m.minimize1d(y, x0, x1, x2, 1e-15)
plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='r')
x0 = 10
x1 = 10.1
x2 = 10.2
x_list, y_list, parabola_points = m.minimize1d(y, x0, x1, x2, 1e-15)
plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='r')
plt.plot(x, y(x), '--', c='grey', alpha=0.4)
plt.xlim([4,9])
plt.ylim([-2,2])
plt.grid()

plt.subplot(2,2,2)
x = np.linspace(-5, 5, 100)
y = lambda x: 0.1*(x+5.5)*(x-4.4)*(x+3.3)
x0 = 10
x1 = 9
x2 = 8
x_list, y_list, parabola_points = m.minimize1d(y, x0, x1, x2, 1e-15)
plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='r')
plt.plot(x, y(x), '--', c='grey', alpha=0.4)
plt.xlim([-5,5])
plt.ylim([-10,3])
plt.grid()

plt.subplot(2,2,3)
x = np.linspace(-5, 5, 100)
y = lambda x: -np.exp(-0.5*x*x)
x0 = -0.6
x1 = -0.5
x2 = -0.4
x_list, y_list, parabola_points = m.minimize1d(y, x0, x1, x2, 1e-15)
plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='r')
plt.plot(x, y(x), '--', c='grey', alpha=0.4)
plt.xlim([-1,1])
plt.ylim([-1,-0.8])
plt.grid()

plt.subplot(2,2,4)
x = np.linspace(0, 10, 100)
y = lambda x: (x-2)*(x-3)*(x-4)*(x-5)*(x-6)
x0 = 2.9
x1 = 3
x2 = 3.1
x_list, y_list, parabola_points = m.minimize1d(y, x0, x1, x2, 1e-15)
plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='r')
x0 = 8
x1 = 9
x2 = 10
x_list, y_list, parabola_points = m.minimize1d(y, x0, x1, x2, 1e-15)
plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='r')
plt.plot(x, y(x), '--', c='grey', alpha=0.4)
plt.xlim([2,6.5])
plt.ylim([-4,4])
plt.grid()

# %%
