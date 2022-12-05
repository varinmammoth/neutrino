#%%
import numpy as np

# %%
def secant(func, x1, x2):
    while True:
        f1 = func(x1)
        f2 = func(x2)
        temp = x1
        x1 = x1-(x1-x2)*f1/(f1-f2)
        x2 = temp
        if abs(f1) < 1e-6:
            return x1

y = lambda x: (x-5)*(x-6)*(x-7)*(x-8)

y_uncer = lambda x: y(x) - (y(5.38)+0.5)

sol1 = secant(y_uncer, 5.38+0.1, 5.38+1.1*0.1)
sol2 = secant(y_uncer, 5.38-0.1, 5.38-1.1*0.1)
# %%
