#!/usr/bin/env python
# coding: utf-8

# In[25]:


import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import time
from numba import jit
import cmath
import scipy.integrate as sci
from scipy.optimize import fsolve



# In[26]:


##########################
#NUMERICAL Integration and Derivatives
@jit(nopython = True)
def simps(f,l,z,a,b,N,*args):
    
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b-a)/N
    if l == 0:
        x = np.linspace(a,b,N+1)
        y = f(x,z,*args)
        S = dx/3 * np.sum(y[0:-1:2] + 4*y[1::2] + y[2::2])
    
    elif l == 1:
        x = np.linspace(a,b,N+1) 
        y = f(z,x,*args)
        S = dx/3 * np.sum(y[0:-1:2] + 4*y[1::2] + y[2::2])
        
    return S

@jit(nopython = True)
def derivative(f,l,x,y,hder,*args):
    if l == 0:
        return ( 0.5*(f(x+np.abs(hder),y,*args)-f(x-np.abs(hder),y,*args))/np.abs(hder) )
    
    elif l == 1:
        return ( 0.5*(f(x,y+np.abs(hder),*args)-f(x,y-np.abs(hder),*args))/np.abs(hder) )
    
    elif (l != 0 and l != 1):
        print("Error. Check 'l' number")
        
##################################


@jit(nopython = True)
def d1(rho,z,M):
    return np.sqrt(rho**2+(z-M)**2)

@jit(nopython = True)
def d2(rho,z,M):
    return np.sqrt(rho**2+(z+M)**2)



@jit(nopython = True)
def xi2(R,z,a):
	return ( np.sqrt(np.abs((a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4))/(R**2 + z**2)))/np.sqrt(2) )


#######################################################################################################


####################
#### Potenciais	####
####################


@jit(nopython = True)
def nuD(R,z,M,a):
	#if (np.abs(z) < 10**-6 and R >= a):
	if xi2(R,z,a) == 0.0:
		#print("a")
		return ( (M*np.sqrt(np.abs(a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))*(-3*a**2 + R**2 + z**2 + 3*np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))/(np.sqrt(2)*np.pi*(R**2 + z**2)**2) - (2*M*(-(a**2*(R**2 - 2*z**2)) + 2*(R**2 + z**2)**2)*np.pi/2)/(np.pi*(R**2 + z**2)**2.5) )
	#if (xi2(R,z,a) == 0.0):
	#	return  (- (2*M*(-(a**2*(R**2 - 2*z**2)) + 2*(R**2 + z**2)**2)*np.pi/2)/(np.pi*(R**2 + z**2)**2.5) )
	else:
		#print("b")
		return ( (M*np.sqrt(np.abs(a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))*(-3*a**2 + R**2 + z**2 + 3*np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))/(np.sqrt(2)*np.pi*(R**2 + z**2)**2) - (2*M*(-(a**2*(R**2 - 2*z**2)) + 2*(R**2 + z**2)**2)*np.arctan(np.sqrt(2)*np.sqrt((R**2 + z**2)/np.abs((a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4))))))/(np.pi*(R**2 + z**2)**2.5) )

@jit(nopython = True)
def nu(rho,z,M,MD,b,m): #Schwarzschild \nu potential: nuSch
    #BH
    #DISK
    	
	nuSch = math.log((d1(rho,z,M) + d2(rho,z,M) - 2*M)/(d1(rho,z,M) + d2(rho,z,M) +2*M))
    
	if m == 0:
        
        	return nuSch
    
	elif m == 1:
  

		return nuD(rho,z,MD,b)
    
	elif m == 2:
		
		return (nuSch + nuD(rho,z,MD,b))



@jit(nopython = True)
def lambSch(rho,z,M,MD,b): #Schwarzshild \lambda potential: lambSch
    sigma = np.sqrt((rho**2 + z**2 + M**2)**2 - 4*z**2*M**2)
    return np.log( ((d1(rho,z,M) + d2(rho,z,M))**2-4*M**2)/(4*sigma) )


@jit(nopython = True)
def gpp(rho,z,M,MD,b):
    return rho**2 * math.exp(- nu(rho,z,M,MD,b,2))



#####################################################################

########      TO CALCULATE THE INITIAL OBSERVER'S RHO       #########

def d1_i(rho,z,M):
    return np.sqrt(rho**2+(z-M)**2)


def d2_i(rho,z,M):
    return np.sqrt(rho**2+(z+M)**2)




def xi2_i(R,z,a):
	return ( np.sqrt((a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4))/(R**2 + z**2))/np.sqrt(2) )



def nuD_i(R,z,M,a):
	#if (np.abs(z) < 10**-6 and R >= a):
	if xi2_i(R,z,a) == 0.0:
		#print("a")
		return ( (M*np.sqrt(np.abs(a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))*(-3*a**2 + R**2 + z**2 + 3*np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))/(np.sqrt(2)*np.pi*(R**2 + z**2)**2) - (2*M*(-(a**2*(R**2 - 2*z**2)) + 2*(R**2 + z**2)**2)*np.pi/2)/(np.pi*(R**2 + z**2)**2.5) )
	#if (xi2(R,z,a) == 0.0):
	#	return  (- (2*M*(-(a**2*(R**2 - 2*z**2)) + 2*(R**2 + z**2)**2)*np.pi/2)/(np.pi*(R**2 + z**2)**2.5) )
	else:
		#print("b")
		return ( (M*np.sqrt(np.abs(a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))*(-3*a**2 + R**2 + z**2 + 3*np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))/(np.sqrt(2)*np.pi*(R**2 + z**2)**2) - (2*M*(-(a**2*(R**2 - 2*z**2)) + 2*(R**2 + z**2)**2)*np.arctan(np.sqrt(2)*np.sqrt((R**2 + z**2)/np.abs((a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4))))))/(np.pi*(R**2 + z**2)**2.5) )


def nu_i(rho,z,M,MD,b,m): #Schwarzschild \nu potential: nuSch
    #BH
    #DISK
    	
	nuSch = math.log((d1_i(rho,z,M) + d2_i(rho,z,M) - 2*M)/(d1_i(rho,z,M) + d2_i(rho,z,M) +2*M))
    
	if m == 0:
        
        	return nuSch
    
	elif m == 1:
  

		return nuD(rho,z,MD,b)
    
	elif m == 2:
		
		return (nuSch + nuD_i(rho,z,MD,b))


def gpp_i(rho,z,M,MD,b):
    return rho**2 * math.exp(- nu_i(rho,z,M,MD,b,2))

###################################################################################
###################################################################################
###################################################################################


    
@jit(nopython = True)
def derNU(rho,z,M,MD,b,l,m,hder):
    return derivative(nu,l,rho,z,hder,M,MD,b,m)
    

@jit(nopython = True)
def dlamb(z,rho,M,MD,b,l,m,hder):
    if l == 0:
        return 0.5*rho * ( derNU(rho,z,M,MD,b,0,m,hder)**2 - derNU(rho,z,M,MD,b,1,m,hder)**2)
    
    elif l == 1:
        return rho * derNU(rho,z,M,MD,b,0,m,hder) * derNU(rho,z,M,MD,b,1,m,hder)

@jit(nopython = True)
def dlamb2(rho,z,M,MD,b,l,m,hder):
    if l == 0:
        return 0.5*rho * ( derNU(rho,z,M,MD,b,0,m,hder)**2 - derNU(rho,z,M,MD,b,1,m,hder)**2)
    
    elif l == 1:
        return rho * derNU(rho,z,M,MD,b,0,m,hder) * derNU(rho,z,M,MD,b,1,m,hder)


@jit
def lamb_Mat(rho,z,M,MD,b,m,hder):
    limit = 0.0
    if z >= limit:
        return sci.quad(dlamb,40.0,z,(rho,M,MD,b,1,m,hder))[0] + sci.quad(dlamb2,40.0,rho,(40.0,M,MD,b,0,m,hder))[0] + lambSch(40.0,40.0,M+MD,MD,b)
    
    elif z < -limit:
        return sci.quad(dlamb,-40,z,(rho,M,MD,b,1,m,hder))[0] + sci.quad(dlamb2,40.0,rho,(-40.0,M,MD,b,0,m,hder))[0] + lambSch(40.0,-40.0,M+MD,MD,b)



"""
@jit
def lamb_Mat(rho,z,M,MD,b,m,hder):
    limit = 0.0
    if z >= limit:
        return sci.quad(dlamb,40.0,z,(rho,M,MD,b,1,m,hder))[0] + sci.quad(dlamb2,40.0,rho,(40.0,M,MD,b,0,m,hder))[0] -2*(M+MD)/np.sqrt(40.0**2+40.0**2)
    
    elif z < -limit:
        return sci.quad(dlamb,-40,z,(rho,M,MD,b,1,m,hder))[0] + sci.quad(dlamb2,40.0,rho,(-40.0,M,MD,b,0,m,hder))[0] -2*(M+MD)/np.sqrt(40.0**2+40.0**2)
"""
#@jit
#def lamb_Mat(rho,z,M,MD,b,m,hder):
#	return sci.quad(dlamb,40.0,z,(rho,M,MD,b,1,m,hder))[0] + sci.quad(dlamb2,40.0,rho,(40.0,M,MD,b,0,m,hder))[0] + lambSch(40.0,40.0,M+MD,MD,b)
   
    

########################################################################################   

#######################################
					#
#CREATE THE NON-LINEAR POTENTIAL MATRIX #
					# 
########################################


z = np.linspace(40.0,-40.0,1200)
rho = np.linspace(40.0,0.0,1200)
nu_Mat = np.zeros((len(z),len(rho)))


count = 0
M = 0.9
MD = 0.1
z0 = 0.0
b = 3.0
func_initial = lambda R : np.sqrt(gpp_i(R,z0,M,MD,b)) - 15.0

R_initial_guess = 10.0
R_solution = fsolve(func_initial, R_initial_guess)

rho0 = float(R_solution) 

for i in range(len(z)):
	for j in range(len(rho)):
		nu_Mat[i,j] = lamb_Mat(rho[j],z[i],M,MD,b,2,10**-6)

for i in range(len(z)):
	for j in range(len(rho)):
		if np.isnan(nu_Mat[i,j]) == True:
			nu_Mat[i,j] = -20
			count += 1
			print(z[i],rho[j])

print(count)
#np.savetxt('Mat_nu_disk_big',nu_Mat)
np.savetxt('Mat_constA_Mbh_0.9',nu_Mat)


########################################################################################
