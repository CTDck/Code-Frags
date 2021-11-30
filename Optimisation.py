# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 09:23:12 2021

@author: Dckenstein
"""

import numpy as np
import matplotlib.pyplot as plt

# Global Variables ##########################################################

wavelength = 1500e-9
dist_to_l2 = 0.3 #Distance to L2 lens from Ref Cavity
dist_to_l1 = 0.3 #Distance to L1 lens from Collimator exit
total_distance = 1.494 #Total distance propagated by the beam
l1_to_l2_dist = total_distance - (dist_to_l1 + dist_to_l2) #Distance between optical elements

desired_waist = 1.868e-3 #Desired waist size in the cavity, for ref 1.868e-3
initial_waist_size = 1e-3 #Initial waist size on collimator exit



class Beam:
    """
    Defines as object beam in various ways:
        
        Method 1 : 1 complex input (a+ib); A beam's Q-value
    
        Method 2 : 2 float inputs, 1 bool, Generates a beams Q-value in
            terms of beam defocus and beam size, bool value is used to 
            differeniate this method from Method 3.
            
        Method 3: 2 float inputs, a waist position and initial waist size 
            determines a beam's Q-value
    """
    def __init__(self,*args):
        if isinstance(args[0], complex):
            self.q = args[0]
        elif len(args) == 3: 
            inverse_q = args[0] + 1j*wavelength/(np.pi*args[1]**2)
            self.q = 1/inverse_q
        else:
            waist_size = args[0]
            waist_position = args[1]
            self.q = waist_position - (1j*np.pi*waist_size**2/wavelength)
    def get_q_value(self):
        return np.array([[self.q,1]]).T
    
    def get_defocus(self):
        return np.real(1/self.q)
    
    def get_beam_size(self):
        return np.sqrt(wavelength/(np.pi*np.imag(1/self.q)))
    
    
def propagate(beam_incident, distance):
    input_q = beam_incident.get_q_value()
    rt_matrix = np.array([[1,distance],[0,1]])
    output = rt_matrix@input_q
    return Beam(complex(output[0]/output[1]))

def gen_waist_function(distance, inital_waist_size, beam):
    return inital_waist_size*np.sqrt(1 + (distance/np.imag(beam.q))**2)

def gen_defoci(distance, initial_waist_size, beam):
    rayleigh_range = np.pi*initial_waist_size**2/wavelength
    roc = distance + rayleigh_range**2/distance
    return 1/roc

def lens(beam_incident, focal_length): #Not used....Redundant...?
    input_q = beam_incident.get_q_value()
    rt_matrix = np.array([[1,0],[-1/focal_length,1]])
    output = rt_matrix@input_q
    return Beam(complex(output[0]/output[1]))

def modematch(beam1, beam2): #Redundant ...?
    q1 = beam1.q
    q2 = beam2.q
    return np.sqrt((q1-np.conj(q1))**2*(q2-np.conj(q2))**2)/(np.abs(q2-np.conj(q1)))**2

#Initalise Beams
collimator_init = Beam(initial_waist_size,0) #Assume origin is at collimator
ref_cavity_init = Beam(desired_waist, total_distance) #Reference cavity beam initialised at full distance between chambers



# Initial Propagation
collimator_init_to_l1 = propagate(collimator_init, dist_to_l1) #Arbitrary choice of L1 position, say .3m along optical path
ref_init_to_l2 = propagate(ref_cavity_init, dist_to_l2) #Arbitrary choice of L2 position, say 1.194m along optical path (total distance 1.494-0.3)

i = 10000
l1_range = 5 # Maximal defocus range for l1
l1_waist = collimator_init_to_l1.get_beam_size() # Get Beam Size at L1
l1_defocus = collimator_init_to_l1.get_defocus() # Get defocus at L1

print("\nWaist Properties at L1...")
print(" Get l1 waist method: ", "   %.6g" % l1_waist + "m")
print(" Expected l1 waist value:", "%.6g" % gen_waist_function(dist_to_l1, initial_waist_size, collimator_init_to_l1) + "m")


#Propagation of tests between L1 and L2
l1_test_defocus = l1_defocus + np.linspace(-l1_range, l1_range, i) #Create an array consisting of copies of defocus at l1
l1_to_l2_test = [] #Empty List to be populated by test propagations with varying defoci
l2_waists = [] #Empty list to be populated with waist at l2 from test cases
dist1 = np.linspace(0, dist_to_l1,1000)
dist2 = np.linspace(0,l1_to_l2_dist,1000)
dist3 = np.linspace(0, dist_to_l2, 1000)

l2_waist_rhs = gen_waist_function(dist3[::-1], desired_waist, ref_init_to_l2)[0]
l2_defocus_rhs = gen_defoci(dist_to_l2, desired_waist, ref_init_to_l2)
difference = 1

for i in range(i):
    l1_init = Beam(l1_waist, l1_test_defocus[i], True) #Generates a test beam with varying defocus which may be iterated over
    test_prop = propagate(l1_init, l1_to_l2_dist) #Propagates the test beam forward
    l2_waist = gen_waist_function(dist2, l1_waist, test_prop)[-1] #Waist at L2 will be the last element in the array
    l1_to_l2_test.append(test_prop) #Adds test case to list of test cases
    l2_waists.append(l2_waist) #Adds waist achieved to a list of waists
    if np.abs(l2_waist_rhs - l2_waist) < difference:
        difference = np.abs(l2_waist_rhs - l2_waist) # Redefine difference if condition is met
        position = i #Note of position of optimal value
        l2_waist_lhs = l2_waist #Optimal value of waist is set
        l2_defocus_lhs = gen_defoci(l1_to_l2_dist, l1_waist, test_prop) #Defocus at a L2 calc'd

# OUTPUTS
#Output Waist properties at L2

print("\nWaist Properties at L2...")
print(" Waist at L2 from LHS:", "%.6g" %  l2_waist_lhs)
print(" Waist at L2 from RHS:", "%.6g" % l2_waist_rhs)

#Output Defocus Values at L1

print("\nDefocus values at L1...")
print(" Get l1 defocus method LHS:", "    %.6g" % l1_defocus)
print(" Expected l1 defocus value LHS:", "%.6g" % gen_defoci(dist_to_l1, initial_waist_size, collimator_init_to_l1))
print(" Defocus of Beam at L1 RHS:", "   %.6g" % l1_test_defocus[position])

# Output Defocus Values at L2

print("\nDefocus values at L2...")
print(" Defocus of beam at L2 from LHS:", "%.6g" % l2_defocus_lhs)
print(" Defocus of beam at L2 from RHS:", "%.6g" % l2_defocus_rhs)

#Misc Values

print("\nMisc. Values...")
print("Difference between L2 LHS and L2 RHS", difference)

# Lens Properties...? LHS + RHS beam defocus...? 

print("\nLens Properties...")
print(" Focal length of L2:", "%.3g" % (1/(l2_defocus_lhs + l2_defocus_rhs)))
print(" Focal length of L1:", "%.3g" % (1/(l1_defocus + l1_test_defocus[position])))


# Visual Aid    
plt.figure(figsize=(15,10))
plt.plot(dist1, gen_waist_function(dist1, initial_waist_size, collimator_init_to_l1), label="Waist Collimator to L1")
plt.plot(dist2+dist_to_l1, gen_waist_function(dist2, l1_waist, l1_to_l2_test[position]), label="Waist from L1 to L2")
plt.plot(dist3 + dist_to_l1 + l1_to_l2_dist, gen_waist_function(dist3[::-1], desired_waist, ref_init_to_l2), label="Waist from Ref to L2")
plt.legend()
plt.grid()
plt.title("Test Beam Propagation")
plt.xlabel("Propagation Distance (m)")
plt.ylabel("Waist Size (m)")
plt.show()

"""
plt.figure(figsize=(15,10))
plt.plot(dist2+0.3, gen_waist_function(dist2, l1_waist, l1_to_l2_test[position]), label="Waist from L1 to L2")
plt.plot(dist1+1.194, gen_waist_function(dist1[::-1], 1.868e-3, ref_init_to_l2), label="Waist from Ref to L2")
plt.legend()
plt.grid()
plt.xlim(1.19,1.2)
plt.ylim(0.00186,0.00187)
plt.title("Test Beam Propagation")
plt.xlabel("Propagation Distance (m)")
plt.ylabel("Waist Size (m)")
plt.show()
"""





