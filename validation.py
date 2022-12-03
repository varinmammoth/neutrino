#%%
'''
This is a file to test the functions created.
'''
import numpy as np
import matplotlib.pyplot as plt
import main4 as m
# %%
'''
Test of 1d parabolic minimiser.
'''
plt.figure(figsize=(10, 6), dpi=80)
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
'''
Test of 2d parabolic minimizer.
'''
plt.figure(figsize=(10, 6), dpi=80)
plt.subplot(1,2,1)
x = np.linspace(-5,5,100)
y = np.linspace(-5,5,100)
f = lambda x,y: x**2 + y**2 
X, Y = np.meshgrid(x, y)
F = f(X,Y)
plt.contourf(X,Y,F,20, cmap='RdGy')
plt.colorbar()
x_list, y_list, p1, p2, p3 = m.minimize2d(f, -2.5, 3)
plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='b')
x_list, y_list, p1, p2, p3 = m.minimize2d(f, -2.5, 3, starting_direction='y')
plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='g')

plt.subplot(1,2,2)
x = np.linspace(-np.pi/2,np.pi/2,100)
y = np.linspace(-np.pi/2,np.pi/2,100)
f = lambda x,y: np.sin(x)**2 + np.sin(y)**2
X, Y = np.meshgrid(x, y)
F = f(X,Y)
plt.contourf(X,Y,F,20, cmap='RdGy')
plt.colorbar()
x_list, y_list, p1, p2, p3 = m.minimize2d(f, -0.75, 0.5)
plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='b')
x_list, y_list, p1, p2, p3 = m.minimize2d(f, 0.4, -0.45, starting_direction='y')
plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='g')


# %%
'''
Test of gradient descent
'''
plt.figure(figsize=(10, 6), dpi=80)
plt.subplot(2,2,1)
x = np.linspace(-5,5,100)
y = np.linspace(-5,5,100)
f = lambda x,y: x**2 + y**2 
X, Y = np.meshgrid(x, y)
F = f(X,Y)
plt.contourf(X,Y,F,20, cmap='RdGy')
plt.colorbar()
x_list, y_list, p1 = m.grad_2d(f, [], -2.5, 3, alpha=0.1)
plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='b')
x_list, y_list, p1 = m.grad_2d(f, [], -2.5, -3, alpha=0.1)
plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='g')

plt.subplot(2,2,2)
x = np.linspace(-np.pi/2,np.pi/2,100)
y = np.linspace(-np.pi/2,np.pi/2,100)
f = lambda x,y: np.sin(x)**2 + np.sin(y)**2
X, Y = np.meshgrid(x, y)
F = f(X,Y)
plt.contourf(X,Y,F,20, cmap='RdGy')
plt.colorbar()
x_list, y_list, p1 = m.grad_2d(f, [], -1.2, 1.3, alpha=0.1)
plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='b')
x_list, y_list, p1 = m.grad_2d(f, [], 1.2, -1.3, alpha=0.1)
plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='g')

plt.subplot(2,2,3)
x = np.linspace(-2,2,100)
y = np.linspace(-2,2,100)
f = lambda x,y: -np.exp(-0.5*x*x - 0.25*y*y)
X, Y = np.meshgrid(x, y)
F = f(X,Y)
plt.contourf(X,Y,F,20, cmap='RdGy')
plt.colorbar()
x_list, y_list, p1 = m.grad_2d(f, [], -1.2, 1.3, alpha=0.1)
plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='b')
x_list, y_list, p1 = m.grad_2d(f, [], 1.2, -1.3, alpha=0.1)
plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='g')

plt.subplot(2,2,4)
x = np.linspace(-10,10,100)
y = np.linspace(-10,10,100)
f = lambda x,y: x**2 + 2*y**2 + x*y + 3*x
X, Y = np.meshgrid(x, y)
F = f(X,Y)
plt.contourf(X,Y,F,20, cmap='RdGy')
plt.colorbar()
x_list, y_list, p1 = m.grad_2d(f, [], -5, -5, alpha=0.1)
plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='b')
x_list, y_list, p1 = m.grad_2d(f, [], 5, 5, alpha=0.1)
plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='g')
# %%
'''
Test of 2d Newton method
'''
plt.figure(figsize=(10, 6), dpi=80)
plt.subplot(1,2,1)
x = np.linspace(-10,10,100)
y = np.linspace(-10,10,100)
f = lambda x,y: x**2 + 2*y**2 + x*y + 3*x
X, Y = np.meshgrid(x, y)
F = f(X,Y)
plt.contourf(X,Y,F,20, cmap='RdGy')
plt.colorbar()
x_list, y_list = m.newton2d(f, 5, 5, 1e-5, 1e-5, max_iterations=20)
plt.quiver(x_list[:-1], y_list[:-1], x_list[1:]-x_list[:-1], y_list[1:]-y_list[:-1], scale_units='xy', angles='xy', scale=1, color='b')

# %%
'''
Test of 3d gradient descent
'''
plt.figure(figsize=(10, 6), dpi=80)
plt.subplot(1,2,1)
f = lambda x,y,z: x**2 + y**2 + z**2 + 5
x_list, y_list, z_list, f_list = m.grad_3d(f, [], 5, 5, 5, alpha=1e-1, delta=1e-3)

plt.plot(f_list, '.')
plt.xlabel('Iteration')
plt.ylabel('f')
plt.grid()
plt.show()
# %%
'''
Test of 3d Newton method
'''
plt.figure(figsize=(10, 6), dpi=80)
plt.subplot(1,2,1)
f = lambda x,y,z: x**2 + y**2 + z**2 + 5
x_list, y_list, z_list = m.newton3d(f, 5, 5, 5, 1e-4, 1e-4, 1e-4)

plt.plot(f(x_list, y_list, z_list), '.')
plt.xlabel('Iteration')
plt.ylabel('f')
plt.grid()
plt.show()
# %%
