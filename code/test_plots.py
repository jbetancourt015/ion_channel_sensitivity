'''
Creates a simple plot.

'''
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(10)
y = np.sin(x)

plt.plot(x,y,c='r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()