# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:11:02 2021

@author: Dckenstein
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def spot_size(initial_waist, x, confocal_parameter):
    return initial_waist*np.sqrt(1+(x/confocal_parameter)**2)

def herm(x,order):
    '''
    Parameters
    ----------
    x : Any input, typically an array of values
    order : Integer specifying the order n of the final hermite polynomial
    Returns
    -------
    Hermite polynomial of order n evaluated at a value x 
    '''
    if order == 0:
        return 1.0 + 0*x
    elif order == 1:
        return 2*x
    else:
        return 2*x*herm(x,order-1)-2*(order-1)*herm(x,order-2)
    
def Gauss_Hermite(x, waist, order):
    '''
    Parameters
    ----------
    x : Any input, typically an array of values
    waist : specifies waist of the gauss hermite function
    order : specifies the degree of hermite polynomial
    Returns
    -------
    Gauss-Hermite function of one variable
    '''
    U_0 = np.exp(-x**2/(waist**2))
    herm_val = np.sqrt(2)*x/waist
    return U_0*herm(herm_val, order)

def Gauss_Hermite_3d(x, y, waist, order1, order2):
    '''
    Parameters
    ----------
    x : Any input, typically an array of values with matching length y.
    y : Any input, typically an array of values with matching length x
    waist : specifies the waist size of the gauss-hermite function
    order1 : specifies the degree of hermite polynomials for x-values
    order2 : specifies the degree of hermite polynomials for y-values
    
    Returns
    -------
    Gauss-Hermite function of two variables, plottable in 3d space
    '''
    a, b = np.meshgrid(x,y)
    U_0 = np.exp(-a**2/(waist**2))
    U_1 = np.exp(-b**2/(waist**2))
    herm_valx = np.sqrt(2)*a/waist
    herm_valy = np.sqrt(2)*b/waist
    return (U_0*U_1*herm(herm_valx,order1)*herm(herm_valy, order2))**2



x = np.linspace(-2,2,1000)
y = np.linspace(-2,2,1000)

X, Y = np.meshgrid(x,y)

plt.figure(figsize=(10,10))
plt.title("Cross-section along x-axis of distribution", fontsize=16)
plt.ylabel("Normalised laser intensity", fontsize=16)
plt.xlabel("Horizontal position relative to optic axis (mm)", fontsize=16)
plt.plot(x,(Gauss_Hermite(x, 1, 0))**2, label="Normalised Intensity")
plt.axvline(1, linestyle="dashed", label="Waist Size")
plt.grid()
plt.legend(fontsize=16)
plt.show()

plt.figure(figsize=(10,10))
ax = plt.axes(projection="3d")
ax.plot_wireframe(X,Y,(Gauss_Hermite_3d(x, y, 1, 0, 0)), cmap="viridis", linewidth=1, antialiased=True, label="Intensity Profile")
ax.set_xlabel("Horizontal Position (mm)", fontsize=16)
ax.set_ylabel("Vertical position (mm)", fontsize=16)
ax.set_zlabel("Normalised Laser Intensity", fontsize=16)
ax.set_title("TEM00 profile exemplar, waist size 1mm", fontsize=16)
ax.legend(fontsize=16)
plt.draw()