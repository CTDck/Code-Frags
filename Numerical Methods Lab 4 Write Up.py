#!/usr/bin/env python
# coding: utf-8

# # An analysis on select Partial Differential Equation solving methods and their respective stabilities

# ## 1.1: Upwind Method
# 
# ### The General Problem
# 
# Consider a distribution at some initial condition. We wish to model the progress of the distribution as a function of time. We can model such a distribution using a 1st order partial differential equation of the form: 
# 
# $$ \frac{\partial u}{\partial t} = -v \frac{\partial u}{\partial x} $$
# 
# With some initial condition describing the distribution at time $t = 0$ (more precisely: $u(x,0)$). The distribution moves with some velocity $v$.
# 
# In this case a Gaussian distribution is the chosen initial condition for the function: 
# 
# $$ u(x,0) = A\exp \left[ \frac{-(x-x_{0})^{2}}{ (2d^{2}) } \right] $$
# 
# Where $x_{0}$ is the mean of the distribution, $d$ is the standard deviation of the distribution, and $A$ is the value at the mean $x_{0}$. We also note that in this instance we have taken $v$, the velocity, to be a constant in both space and time.
# 
# The Upwind method may be used here provided the stability criteria is met. The Upwind scheme takes the form:
# 
# 
# $$ \frac{u_{j}^{n+1} - u_{j}^{n}}{\Delta t} = -v_{j}^{n} \begin{cases} \frac{u_{j}^{n} - u_{j-1}^{n}}{\Delta x} & v_{j}^{n} > 0 \\ \frac{u_{j+1}^{n} - u_{j}^{n}}{\Delta x} & v_{j}^{n} < 0 \end{cases} $$
# 
# For the case above $v$ is taken to be positive. This implies for all $v$:
# 
# $$ \frac{u_{j}^{n+1} - u_{j}^{n}}{\Delta t} = -v_{j}^{n} \frac{u_{j}^{n} - u_{j-1}^{n}}{\Delta x} $$
# 
# Once rearranged, the above becomes:
# 
# $$ u_{j}^{n+1} = u_{j}^{n} - v_{j}^{n} \frac{\Delta t}{\Delta x} (u_{j}^{n} - u_{j-1}^{n})$$
# 
# This is the basis for coding an Upwind method function. Below is the pseudocode and subsequent function to carry out the upwind method given the initial conditions.

# In[2]:


# -----------------------------------------------------------------
#
#function(xmax,tmax,dx,dt,cutoff):
#    set x = array of x values incremented by dx
#    set t = array of t values incremented by dt
#    set u = array of zeroes for solutions at time tn
#    set u_a = array of zeroes for analytic solutions at time tn
#    
#    do for all t:
#        do for all x:
#            add analytic solution at tn,xn to u_a
#            if t is first increment:
#                do u_initial
#            else:
#                do u_n+1j = u_nj - v(dt/(dx))(u_nj-u_nj-1)
#    do plotfunction at cutoff
#
# -----------------------------------------------------------------


# In[6]:


# necessary imports

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import Image as Image2
from IPython.display import display


# init variables
A = 10
x0 = 10
d = 1.5
v = 2

def Upwind_Method_animated(xmax,tmax,dx,dt,cutoff): # <~ function definition blah blah, change to include xmin and tmin!
    """
    Args:
        xmax: Maximal X value
        tmax: Maximal time value
        cutoff: how many solutions you wish to show animated
        dx: increment of x
        dt: increment of time
    Returns:
        An animated gif of the numerical solutions given by the upwind method and analytical solutions at set time increments
    """
    x = np.arange(0,xmax,dx)
    t = np.arange(0,tmax,dt)
    x_space = int(xmax/dx)
    t_space = int(tmax/dt)
    
    u = np.zeros([t_space,x_space]) # <~ creates a zero array of t rows in x columns
    u_analytic = np.zeros([t_space,x_space]) # <~ as above but for the analytic solution
    
    for time in range(t_space-1): # <~ you could use for or while loops here, just change a few things
        for position in range(x_space-1):
            u_analytic[time,position] = A*np.exp(-(x[position]-x0-v*t[time])**2/(2*d**2))
            if time == 0: # <~ initial condition evaluation
                u[time,position] = A*np.exp(-(x[position]-x0)**2/(2*d**2))
            else: # <~ everything else follows the equation under the upwind method
                u[time,position] = u[time-1,position] - v*(dt/dx)*(u[time-1,position]-u[time-1,position-1]) # <~ said equation in code form
    # plots graphs across all cutoff sections and animates to a .gif file
    i = 0
    frames = []
    while i < cutoff:
        plt.figure(figsize=(10,10))
        plt.plot(x,u[i,:], label="Upwind Numerical Solution") # <~ plots numerical solution
        plt.plot(x,u_analytic[i,:], label="Analytic Solution") #  <~ plots analytic solution (the actual one)
        plt.xlabel("X")
        plt.ylabel("U(X,T)")
        plt.title("Upwind Method compared with Analytic Solution") 
        plt.ylim(-2,12)
        plt.legend()
        canvas = plt.get_current_fig_manager().canvas
        plt.close()
        canvas.draw()
        im = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        frames.append(im)
        i += 1
    frames[0].save("UpwindMethod.gif", format="gif", append_images=frames[1:], save_all=True, duration=20, loop=0)
    display(Image2(filename="UpwindMethod.gif"))
    
Upwind_Method_animated(60,20,0.2,0.1,199) # <~ function call to show it works


# In[7]:


def Upwind_Method(xmax,tmax,dx,dt,cutoff): # <~ function definition blah blah, change to include xmin and tmin!
    """
    Args:
        xmax: Maximal X value
        tmax: Maximal time value
        cutoff: how many solutions you wish to show animated
        dx: increment of x
        dt: increment of time
    Returns:
        A plot of the numerical solution given by the Upwind method and analytical solution at a given time increment
    """
    x = np.arange(0,xmax,dx)
    t = np.arange(0,tmax,dt)
    x_space = int(xmax/dx)
    t_space = int(tmax/dt)
    
    u = np.zeros([t_space,x_space]) # <~ creates a zero array of t rows in x columns
    u_analytic = np.zeros([t_space,x_space]) # <~ as above but for the analytic solution
    
    for time in range(t_space-1): # <~ you could use for or while loops here, just change a few things
        for position in range(x_space-1):
            u_analytic[time,position] = A*np.exp(-(x[position]-x0-v*t[time])**2/(2*d**2))
            if time == 0: # <~ initial condition evaluation
                u[time,position] = A*np.exp(-(x[position]-x0)**2/(2*d**2))
            else: # <~ everything else follows the equation under the upwind method
                u[time,position] = u[time-1,position] - v*(dt/dx)*(u[time-1,position]-u[time-1,position-1]) # <~ said equation in code form
                
    plt.figure(figsize=(15,15))
    plt.plot(x,u[cutoff,:], label="Upwind Numerical Solution") # <~ plots numerical solution
    plt.plot(x,u_analytic[cutoff,:], label="Analytic Solution") # plots analytical solution
    plt.xlabel("X")
    plt.ylabel("U(X,T)")
    plt.title("Upwind Method compared with Analytic Solution")
    plt.ylim(0,12)
    plt.legend()
    plt.show()
    
Upwind_Method(60,20,0.2,0.1,100)


# The Numerical and Analytical solution are in agreement over the timespan of 20 seconds as seen above. Due to the stability criterion being met the numerical solution is stable over the time range. If the stability condition was not met, for example if the $v>2$ the numerical solution would become unstable. 

# ## 1.2: Upwind Method Stability
# 
# The stability of the Upwind scheme is determined by the Courant condition. This is given by:
# 
# $$ \frac{v\Delta t}{\Delta x} \leq 1 $$
# 
# Provided this condition is adhered to the upwind method should be stable. If this condition equals one then the solution is at its most stable state. If this condition is violated then the solution becomes unstable. 
# 
# To illustrate this instability when the Courant condition is violated let $v = 4$.

# In[4]:


v = 4

Upwind_Method_animated(60,20,0.2,0.1,125)


# It does not take long for the numerical solution to effectively explode compared to the analytical solution. This is due to the violation of the Courant condition. To illustrate the instabilities effect on the solution numerical solutions at $t = 0.1, t = 0.5, t = 1.0$ and $t=2.0$ were constructed

# In[37]:


def Upwind_Method_instability_demo(xmax,tmax,dx,dt,cutoff): # <~ function definition blah blah, change to include xmin and tmin!
    """
    Args:
        xmax: Maximal X value
        tmax: Maximal time value
        cutoff: list of which increments in time you would like to print out the solution at
        dx: increment of x
        dt: increment of time
    Returns:
        The plot of the numerical solution given by the Upwind method and analytical solution at the given cutoff points
    """
    x = np.arange(0,xmax,dx)
    t = np.arange(0,tmax,dt)
    x_space = int(xmax/dx)
    t_space = int(tmax/dt)
    
    u = np.zeros([t_space,x_space]) # <~ creates a zero array of t rows in x columns
    u_analytic = np.zeros([t_space,x_space]) # <~ as above but for the analytic solution
    
    for time in range(t_space-1):
        for position in range(x_space-1):
            u_analytic[time,position] = A*np.exp(-(x[position]-x0-v*t[time])**2/(2*d**2))
            if time == 0: # <~ initial condition evaluation
                u[time,position] = A*np.exp(-(x[position]-x0)**2/(2*d**2))
            else: # <~ everything else follows the equation under the upwind method
                u[time,position] = u[time-1,position] - v*(dt/dx)*(u[time-1,position]-u[time-1,position-1]) # <~ said equation in code form
    i = 0
    plt.figure(figsize=(15,15))
    while i < len(cutoff):
        plt.subplot(2,2,i+1)
        plt.plot(x,u[cutoff[i],:], label="Upwind Numerical Solution") # <~ plots numerical solution
        plt.plot(x,u_analytic[cutoff[i],:], label="Analytic Solution") # plots analytical solution
        plt.xlabel("X")
        plt.ylabel("U(X,T)")
        plt.title("Upwind Method compared with Analytic Solution at t=" + str(cutoff[i]*dt))
        plt.ylim(0,12)
        plt.legend()
        plt.grid()
        i += 1
    plt.tight_layout()
    plt.show()


# In[38]:


v = 4
Upwind_Method_instability_demo(60,20,0.2,0.1,[1,5,10,20])


# In[30]:


Upwind_Method_animated(60,20,0.2,0.05,200)


# In[69]:


v = 2
Upwind_Method_animated(60,20,0.2,0.01,400)


# Decreasing the time step leads to the peak of the numerical solution falling and base of the solution getting wider as time progresses. The solution is semi stable but not as stable as it could be if the Courant condition was met. This would imply that the algorithm has to calculate a wider array of non-zero terms leading to the program running slower than expected.

# In[15]:


import time # <~ module used to measure the runtime (time it takes to run a program) of the function setup

v = 2

def Upwind_Method_runtime(xmax,tmax,dx,dt,cutoff): # <~ function definition
    """
    Args:
        xmax: Maximal X value
        tmax: Maximal time value
        cutoff: which increment in time you would like to print out the solution at
        dx: increment of x
        dt: increment of time
    Returns:
        The numerical solution given by the Upwind method at the given cutoff point
    """
    x = np.arange(0,xmax,dx)
    t = np.arange(0,tmax,dt)
    x_space = int(xmax/dx)
    t_space = int(tmax/dt)
    
    u = np.zeros([t_space,x_space]) # <~ creates a zero array of t rows in x columns
    u_analytic = np.zeros([t_space,x_space]) # <~ as above but for the analytic solution
    
    for time in range(t_space-1):
        for position in range(x_space-1):
            u_analytic[time,position] = A*np.exp(-(x[position]-x0-v*t[time])**2/(2*d**2))
            if time == 0: # <~ initial condition evaluation
                u[time,position] = A*np.exp(-(x[position]-x0)**2/(2*d**2))
            else: # <~ everything else follows the equation under the upwind method
                u[time,position] = u[time-1,position] - v*(dt/dx)*(u[time-1,position]-u[time-1,position-1]) # <~ said equation in code form
    return u[cutoff,:] # <~ plots numerical solution
    
def test(dx,dt):
    """
    Args:
        dx: increment of x
        dt: increment of time
    Returns:
        The numerical solution at a cutoff of 100 
    """
    Upwind_Method_runtime(60,20,dx,dt,100)
    
start_time = time.time() # <~ initialise starting time
test(0.02,0.004)
value1 = time.time() - start_time

start_time = time.time()
test(0.01,0.002)
value2 = time.time() - start_time

print(value1,value2) # <~ prints time taken to execute test 1 and test 2 respectively


# As predicted a smaller time increment does increase the runtime of the program.

# ## 2.1 FTCS Method
# 
# It is essential that there is a second order accuracy in time. Without it our the numerical solution very poor regardless of the variables. Using the forward time, centred space method or FTCS method this will prove a second order accuracy in time is required.  The space differential now takes the form: 
# 
# $$ \frac{u_{j}^{n+1} - u_{j}^{n}}{\Delta t} = -v \frac{u_{j+1}^{n} - u_{j-1}^{n}}{2\Delta x} $$
# 
# This is inherently unstable, there is no stable condition for this method.
# 
# To validate this claim the Courant condition was set to 1. Let $dt = 0.1$ and $dx = 0.2$ with $v = 2$

# In[ ]:


# -----------------------------------------------------------------
#
#function(xmax,tmax,dx,dt,cutoff):
#    set x = array of x values incremented by dx
#    set t = array of t values incremented by dt
#    set u = array of zeroes for solutions at time tn
#    set u_a = array of zeroes for analytic solutions at time tn
#    
#    do for all t:
#        do for all x:
#            add analytic solution at tn,xn to u_a
#            if t is first increment:
#                do u_initial
#            else:
#                do u_n+1j = u_nj - v(dt/(2dx))(u_nj+1-u_nj-1)
#    do plotfunction at cutoff
#
# -----------------------------------------------------------------


# In[39]:


def FTCS_Method_animated(xmax, xmin, tmin, tmax, cutoff, dx, dt):
    """
    Args:
        xmax: Maximal X value
        xmin: Minimal X value
        tmin: Minimal time value
        tmax: Maximal time value
        cutoff: how many solutions you wish to show animated
        dx: increment of x
        dt: increment of time
    Returns:
        An animated gif of the numerical and analytical solutions at set time increments
    """
    x = np.arange(xmin,xmax,dx) 
    t = np.arange(tmin,tmax,dt)
    x_space = int(xmax/dx)
    t_space = int(tmax/dt)
    u = np.zeros([t_space,x_space])
    u_analytic = np.zeros([t_space,x_space])
    
    for time in range(t_space-1):
        for position in range(x_space-1):
            u_analytic[time,position] = A*np.exp(-(x[position]-x0-v*t[time])**2/(2*d**2))
            if time == 0:
                u[time,position] = A*np.exp(-(x[position]-x0)**2/(2*d**2))
            else:
                u[time,position] = u[time-1,position] - v*(dt/(2*dx))*(u[time-1,position+1]-u[time-1,position-1]) # exactly as upwind method except we have 2*dx and u_nj+1
    # creates animated gif        
    i = 0
    frames = []
    while i < cutoff:
        plt.figure(figsize=(10,10))
        plt.plot(x,u[i,:], label="FTCS Numerical Solution") # <~ plots numerical solution
        plt.plot(x,u_analytic[i,:], label="Analytic Solution") # <~ plots analytic solution
        plt.xlabel("X")
        plt.ylabel("U(X,T)")
        plt.title("FTCS Method compared with Analytic Solution")
        plt.ylim(-2,12)
        plt.legend()
        canvas = plt.get_current_fig_manager().canvas
        plt.close()
        canvas.draw()
        im = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        frames.append(im)
        i += 1
    frames[0].save("FTCSMethod.gif", format="gif", append_images=frames[1:], save_all=True, duration=20, loop=0)
    display(Image2(filename="FTCSMethod.gif"))


# In[43]:


## comments as above

def FTCS_Method(xmax, xmin, tmin, tmax, cutoff, dx, dt):
    """
    Args:
        xmax: Maximal X value
        xmin: Minimal X value
        tmin: Minimal time value
        tmax: Maximal time value
        cutoff: which increment in time you would like to print out the solution at
        dx: increment of x
        dt: increment of time
    Returns:
        The numerical solution given by the FTCS method and analytical solution at the a given cutoff point
    """
    x = np.arange(xmin,xmax,dx) 
    t = np.arange(tmin,tmax,dt)
    x_space = int(xmax/dx)
    t_space = int(tmax/dt)
    u = np.zeros([t_space,x_space])
    u_analytic = np.zeros([t_space,x_space])
    
    for time in range(t_space-1):
        for position in range(x_space-1):
            u_analytic[time,position] = A*np.exp(-(x[position]-x0-v*t[time])**2/(2*d**2))
            if time == 0:
                u[time,position] = A*np.exp(-(x[position]-x0)**2/(2*d**2))
            else:
                u[time,position] = u[time-1,position] - v*(dt/(2*dx))*(u[time-1,position+1]-u[time-1,position-1])
    # method plots a static figure at a specific cutoff       
    plt.figure(figsize=(15,15))
    plt.plot(x,u[cutoff,:], label="FTCS Numerical Solution")
    plt.plot(x,u_analytic[cutoff,:], label="Analytic Solution")
    plt.xlabel("X")
    plt.ylabel("U(X,T)")
    plt.title("FTCS Method compared with Analytic Solution") 
    plt.ylim(-2,12)
    plt.legend()
    plt.show()


# In[41]:


v = 2
FTCS_Method_animated(60,0,0,20,199,0.2,0.1) # <~ play around with cutoff and watch the soln explode


# In[44]:


FTCS_Method(60,0,0,20,50,0.2,0.1)


# As evident from the above animation it can been seen that the FTCS Numerical solution is unstable even at the Courant condition. This method is therefore deemed obsolete and shows a dependancy on second order accuracy in time.

# ## 2.2: Staggered Leapfrog Method
# 
# The time derivative of the FTCS method may be altered to give second order accuracy in both space and time. This gives rise to the staggered leapfrog method:
# 
# $$ \frac{u_{j}^{n+1} - u_{j}^{n-1}}{2\Delta t} = -v \frac{u_{j+1}^{n} - u_{j-1}^{n}}{2\Delta x} $$
# 
# This method requires the programmer to keep the solution at $n-1$ in order to evaluate the solution at $n+1$. To verify that the method works as intended the initial test conditions for the upwind method were supplied. Below is the pseudocode and subsequent function generated to solve the PDE via the staggered leapfrog method.

# In[1]:


# -----------------------------------------------------------------
#
#function(xmax,tmax,dx,dt,cutoff):
#    set x = array of x values incremented by dx
#    set t = array of t values incremented by dt
#    set u = array of zeroes for solutions at time tn
#    set u_a = array of zeroes for analytic solutions at time tn
#    
#    do for all t:
#        do for all x:
#            add analytic solution at tn,xn to u_a
#            if t is first increment:
#                do u_initial
#            elif t is second increment:
#                do u_n+1j = u_nj - v(dt/dx)(u_nj+1-u_nj-1)
#            else:
#                do u_n+1j = u_n-1j - v(dt/dx)(u_nj+1-u_nj-1)
#    do plotfunction at cutoff
#
# -----------------------------------------------------------------


# In[49]:


def StaggerredLeapFrog_Method_animated(xmax, xmin, tmin, tmax, cutoff, dx, dt):
    """
    Args:
        xmax: Maximal X value
        xmin: Minimal X value
        tmin: Minimal time value
        tmax: Maximal time value
        cutoff: how many solutions you wish to show animated
        dx: increment of x
        dt: increment of time
    Returns:
        An animated gif of the numerical solutions given by the staggered leapfrog method and analytical solutions at set time increments
    """
    x = np.arange(xmin,xmax,dx) 
    t = np.arange(tmin,tmax,dt)
    x_space = int(xmax/dx)
    t_space = int(tmax/dt)
    u = np.zeros([t_space,x_space])
    u_analytic = np.zeros([t_space,x_space])
    
    for time in range(t_space):
        for position in range(x_space-1):
            u_analytic[time,position] = A*np.exp(-(x[position]-x0-v*t[time])**2/(2*d**2))
            if time == 0:
                u[time,position] = A*np.exp(-(x[position]-x0)**2/(2*d**2))
            elif time == 1: #works out solution at first time increment to allow use of stag. leapfrog method
                u[time,position] = u[time-1,position] - v*(dt/dx)*(u[time-1,position+1]-u[time-1,position-1])
            else:
                u[time,position] = u[time-2,position] - v*(dt/dx)*(u[time-1,position+1]-u[time-1,position-1])  # uses the staggered leapfrog method
    # creates animated gif
    i = 0
    frames = []
    while i < cutoff:
        plt.figure(figsize=(10,10))
        plt.plot(x,u[i,:], label="Staggered Leapfrog Numerical Solution") # <~ plots numerical solution
        plt.plot(x,u_analytic[i,:], label="Analytic Solution")
        plt.xlabel("X")
        plt.ylabel("U(X,T)")# <~ plots analytic solution (the actual one)
        plt.title("Staggered Leapfrog Method compared with Analytic Solution") # <~ blah blah add titles and labels as required
        plt.ylim(-2,12)
        plt.legend()
        canvas = plt.get_current_fig_manager().canvas
        plt.close()
        canvas.draw()
        im = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        frames.append(im)
        i += 1
    frames[0].save("LeapfrogMethod.gif", format="gif", append_images=frames[1:], save_all=True, duration=20, loop=0)
    display(Image2(filename="LeapfrogMethod.gif"))


# In[8]:


def StaggerredLeapFrog_Method(xmax, xmin, tmin, tmax, cutoff, dx, dt):
    """
    Args:
        xmax: Maximal X value
        xmin: Minimal X value
        tmin: Minimal time value
        tmax: Maximal time value
        cutoff: which increment in time you would like to print out the solution at
        dx: increment of x
        dt: increment of time
    Returns:
        The numerical solution given by the staggered leapfrog method and analytical solution at the a given cutoff point
    """
    x = np.arange(xmin,xmax,dx) 
    t = np.arange(tmin,tmax,dt)
    x_space = int(xmax/dx)
    t_space = int(tmax/dt)
    u = np.zeros([t_space,x_space])
    u_analytic = np.zeros([t_space,x_space])
    
    for time in range(t_space):
        for position in range(x_space-1):
            u_analytic[time,position] = A*np.exp(-(x[position]-x0-v*t[time])**2/(2*d**2))
            if time == 0:
                u[time,position] = A*np.exp(-(x[position]-x0)**2/(2*d**2))
            elif time == 1:
                u[time,position] = u[time-1,position] - v*(dt/dx)*(u[time-1,position+1]-u[time-1,position-1])
            else:
                u[time,position] = u[time-2,position] - v*(dt/dx)*(u[time-1,position+1]-u[time-1,position-1])  
    plt.figure(figsize=(15,15))
    plt.plot(x,u[cutoff,:], label="Staggered Leapfrog Numerical Solution") # <~ plots numerical solution
    plt.plot(x,u_analytic[cutoff,:], label="Analytic Solution")
    plt.xlabel("X")
    plt.ylabel("U(X,T)")# <~ plots analytic solution (the actual one)
    plt.title("Staggered Leapfrog Method compared with Analytic Solution") # <~ blah blah add titles and labels as required
    plt.ylim(-2,12)
    plt.show()


# In[45]:


v = 2
StaggerredLeapFrog_Method_animated(60, 0, 0, 20, 199, 0.2, 0.1)


# From the above it is evident that the method follows the analytical solution in a local region. Outwith this local region a minor anomaly is present. This method can be deemed stable for the perturbation within the analytical solution but unstable for the entirety of the analytical solution.
# 
# The staggered leapfrog method will tend to be unreliable at the $t=20$ if the Courant Condition is not adhered to. This may be demonstrated by letting $dx=0.4$.

# In[10]:


v = 2
StaggerredLeapFrog_Method(60, 0, 0, 20, 100, 0.4, 0.1)


# In[11]:


StaggerredLeapFrog_Method(60, 0, 0, 20, 199, 0.4, 0.1)


# It can be seen that for a change in $dx$ with the Courant condition violated, the numerical solution still approximates the analytical solution within a local region. This fails at higher and higher values for time.
# 
# To make this method more stable at the higher values for time the value of the Courant condition was decreased. While keeping $v = 2$, let $dx = 0.3$ and $dt = 0.001$.

# In[12]:


StaggerredLeapFrog_Method(60, 0, 0, 20, 19999, 0.3, 0.001)


# It can be seen from the above that with the Courant condition value being less than 1 the solution becomes more stable than before at the higher values of time. There is still and error present, this is likely due to the fact that the method still is a finite difference scheme and thus it must be resolved to a required precision. In this case the order of $dx$  and $dt$ were decreased by 1.

# In[14]:


StaggerredLeapFrog_Method(60, 0, 0, 20, 199999, 0.03, 0.0001)


# As shown from the above the system is now more stable at further and further increments of time however this solution has a extremely long runtime. For a larger runtime the system will become more and more stable for a smaller order of $dx$ and $dt$

# In[ ]:




