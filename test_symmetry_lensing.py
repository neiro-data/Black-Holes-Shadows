#!/usr/bin/env python
# coding: utf-8

# In[25]:


import math
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.colors
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


#########             Inverted MORGAN MORGAN DISCS         ###############

"""@jit(nopython = True)
def xx(rho,z,b):
	arg = (-b**2 + rho**2 + z**2 + np.sqrt(4*b**2 * z**2 + (-b**2 + rho**2 + z**2)**2))/ (2*b**2)
	if arg < 0:
		return 0
	else:
		return np.sqrt(arg)

@jit(nopython = True)
def yy(rho,z,b):
    if xx(rho,z,b) == 0:
        return 0
    else:
        return z/(b*xx(rho,z,b))
"""




################## SCHWARZSCHILD BH + DISK case ###################     

# l = 0 corresponds to derivative in respect to RHO
# l = 1 corresponds to derivative in respect to Z
# m = 0 corresponds to nuSch
# m = 1 corresponds to nuD

##################################################################### 


"""
@jit(nopython = True)
def nu(rho,z,M,MD,b,m): #Schwarzschild \nu potential: nuSch
    #BH
    #DISK
    	
	if (rho < 10**-3 and np.abs(z) < 10**-3):
		x1 = 6000.0 
		y1 = np.sign(z)
	else:

		x1 = xx(b**2*rho/(rho**2+z**2),b**2*z/(rho**2+z**2),b)
		y1 = yy(b**2*rho/(rho**2+z**2),b**2*z/(rho**2+z**2),b)
	
	
	nuSch = math.log((d1(rho,z,M) + d2(rho,z,M) - 2*M)/(d1(rho,z,M) + d2(rho,z,M) +2*M))
    
	if m == 0:
        
        	return nuSch
    
	elif m == 1:
        
        	if np.abs(x1) <= 10**-3:
        		nuD = -2*MD/np.sqrt(rho**2 + z**2) * (np.pi/2 + 0.25*( (3*x1**2 +1) * np.pi/2 -3*x1) * (3 * y1**2 - 1) )

        	else:
        		nuD = -2*MD/np.sqrt(rho**2 + z**2) * (np.arctan(1/x1) + 0.25*( (3*x1**2 +1) * np.arctan(1/x1) -3*x1) * (3 * y1**2 - 1) )

        	return nuD
    
	elif m == 2:

        	if np.abs(x1) <= 10**-3:
            		nuD = -2*MD/np.sqrt(rho**2 + z**2) * (np.pi/2 + 0.25*( (3*x1**2 +1) * np.pi/2 -3*x1) * (3 * y1**2 - 1) )
        	else:
             		nuD = -2*MD/np.sqrt(rho**2 + z**2) * (np.arctan(1/x1) + 0.25*( (3*x1**2 +1) * np.arctan(1/x1) -3*x1) * (3 * y1**2 - 1) )
        
        	return (nuSch + nuD)
"""
"""
@jit(nopython = True)
def nuD(rho,z,MD,b):
	s = cmath.sqrt(rho**2 + ( (rho**2+z**2) / b*1j-z)**2 )
	mu = -z + (rho**2+z**2) / b *1j + s
	return ( -3*MD*b**2/ (rho**2+z**2)**(5/2) * ( (z**2 - 0.5*rho**2 + (rho**2+z**2)**2/b**2)*cmath.log(mu) + 0.5 * (3*z + (rho**2+z**2)/b*1j) * s ).imag )
"""

@jit(nopython = True)
def xi2(R,z,a):
	return ( np.sqrt((a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4))/(R**2 + z**2))/np.sqrt(2) )


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


#####################################################################


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


#####################################################################
#####################################################################
#####################################################################


    
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


"""@jit
def lamb_Mat(rho,z,M,MD,b,m,hder):
    limit = 0.0
    if z >= limit:
        return sci.quad(dlamb,40.0,z,(rho,M,MD,b,1,m,hder))[0] + sci.quad(dlamb2,40.0,rho,(40.0,M,MD,b,0,m,hder))[0] + lambSch(40.0,40.0,M+MD,MD,b)
    
    elif z < -limit:
        return sci.quad(dlamb,-40,z,(rho,M,MD,b,1,m,hder))[0] + sci.quad(dlamb2,40.0,rho,(-40.0,M,MD,b,0,m,hder))[0] + lambSch(40.0,-40.0,M+MD,MD,b)

"""
@jit(nopython = True)    
def lamb(rho,z,M,MD,b,m):
 
	zref = 40.0 ; Rref = 40.0

	iR = (zref-z)/zref * (len(Mat_nu)-1)/2 
	jR = (Rref-rho)/(Rref-0.0)*(len(Mat_nu[0])-1)
	

	i0 = int( math.floor( (zref-z)/zref * (len(Mat_nu)-1)/2 ) )  
	j0 = int( math.floor( (Rref-rho)/(Rref-0.0)*(len(Mat_nu[0])-1) ) )


	if np.abs(jR) > len(Mat_nu[0])-1:
		iR = int(len(Mat_nu)/2)+0.5 ; jR = len(Mat_nu[0])-0.5
		i0 = int(math.floor(iR)); j0 = int(math.floor(jR))

	I = iR - i0 
	J = jR - j0

	#print(iR,"",jR,"",i0,"",j0)

	f00 = Mat_nu[i0 , j0] ; f01 = Mat_nu[i0 , j0+1] 
	f10 = Mat_nu[i0+1 , j0] ; f11 = Mat_nu[i0+1 , j0+1]

	return ( (f00 + (f01 - f00)*J) + (f10 -f00  + (f11 + f00 - f01 - f10)*J) * I  )

   
########################################################################################   

#######################################
					#
#CREATE THE NON-LINEAR POTENTIAL MATRIX #
					# 
########################################
"""
z = np.linspace(40,-40,1200)
rho = np.linspace(40,0,1200)
nu_Mat = np.zeros((len(z),len(rho)))

for i in range(len(z)):
	for j in range(len(rho)):
		nu_Mat[i,j] = lamb_Mat(rho[j],z[i],1.0,1.0,6.0,2,10**-6)

for i in range(len(z)):
	for j in range(len(rho)):
		if np.isnan(nu_Mat[i,j]) == True:
			nu_Mat[i,j] = -20
		else:
			continue


np.savetxt('Mat_nu_disk',nu_Mat)

"""
########################################################################################



Mat_nu = np.loadtxt("Mat_nu_disk0.1")


# In[32]:




@jit(nopython = True)
def gtt(rho,z,M,MD,b):
    return -math.exp(nu(rho,z,M,MD,b,2))



@jit(nopython = True)
def grr(rho,z,M,MD,b):
    return math.exp( lamb(rho,z,M,MD,b,2) - nu(rho,z,M,MD,b,2) ) 

@jit(nopython = True)
def gzz(rho,z,M,MD,b):
    return math.exp( lamb(rho,z,M,MD,b,2) - nu(rho,z,M,MD,b,2) ) 

@jit(nopython = True)
def gpp(rho,z,M,MD,b):
    return rho**2 * math.exp(- nu(rho,z,M,MD,b,2))

##########################

@jit(nopython = True)
def zeta(rho,z,M,MD,b):
    return np.sqrt(-1/gtt(rho,z,M,MD,b))

###############

@jit(nopython = True)
def dthe(rho,z,M,MD,b,alfa):
    return 1/np.sqrt(gzz(rho,z,M,MD,b))*np.sin(alfa)

@jit(nopython = True)
def dr(rho,z,M,MD,b,alfa,beta):
    return 1/np.sqrt(grr(rho,z,M,MD,b))*np.cos(alfa)*np.cos(beta)

@jit(nopython = True)
def Pphi(rho,z,M,MD,b,alfa,beta):
    return np.sqrt(gpp(rho,z,M,MD,b))*np.sin(beta)*np.cos(alfa)

@jit(nopython = True)
def Pt(rho,z,M,MD,b,alfa,beta):
    return -1/zeta(rho,z,M,MD,b)

@jit(nopython = True)
def dphi(rho,z,M,MD,b,alfa,beta,p_phi):
    return 1/gpp(rho,z,M,MD,b)*p_phi

@jit(nopython = True)
def dt(rho,z,M,MD,b,alfa,beta,p_t):
    return 1/gtt(rho,z,M,MD,b)*p_t


# In[33]:


@jit(nopython = True)
def run_kut4_mod(F,x,y,h,*args):
    
	hmax = -0.04
	hmin = -10**-7	
	tol = 10**-4
	
	K1 = h * F(x,y,*args)
	K2 = h * F(x + h/4, y + K1/4,*args)
	K3 = h * F(x + 3/8*h, y + 3/32*K1 + 9/32*K2,*args)
	K4 = h * F(x + 12/13*h, y + 1932/2197*K1 - 7200/2197*K2 + 7296/2197*K3, *args)
	K5 = h * F(x + h, y + 439/216*K1 - 8*K2 + 3680/513*K3 - 845/4104*K4, *args)
	K6 = h * F(x + h/2, y -8/27*K1 +2*K2 -3544/2565*K3 + 1859/4104*K4 -11/40*K5, *args) 

	y4 = y + 25/216*K1 + 1408/2565*K3 + 2197/4101*K4 -K5/5 
	y5 = y + 16/135*K1 + 6656/12825*K3 + 28561/56430*K4 - 9/50*K5 + 2/55*K6
	#if (np.abs(y4[2]) < 0.04 or np.abs(y5[2]) < 0.04):
	#if np.abs(y[2]) < 0.04:
	#	hmax = -10**-3
	#else:
	#	hmax = -0.04
	
	error = np.linalg.norm(y5-y4)
	delta = pow(1.0/2.0,(1.0/4.0)) * pow(tol / error, (1.0/4.0))

	"""
	if error > tol:
		it = 0

		while error > tol:
			h = delta * h ; it += 1


			K1 = h * F(x,y,*args)
			K2 = h * F(x + h/4, y + K1/4,*args)
			K3 = h * F(x + 3/8*h, y + 3/32*K1 + 9/32*K2,*args)
			K4 = h * F(x + 12/13*h, y + 1932/2197*K1 - 7200/2197*K2 + 7296/2197*K3, *args)
			K5 = h * F(x + h, y + 439/216*K1 - 8*K2 + 3680/513*K3 - 845/4104*K4, *args)
			K6 = h * F(x + h/2, y -8/27*K1 +2*K2 -3544/2565*K3 + 1859/4104*K4 -11/40*K5, *args) 

			y4 = y + 25/216*K1 + 1408/2565*K3 + 2197/4101*K4 -K5/5
			y5 = y + 16/135*K1 + 6656/12825*K3 + 28561/56430*K4 - 9/50*K5 + 2/55*K6
			
			error = np.linalg.norm(y5-y4)
			delta = pow(1.0/2.0,(1.0/4.0)) * pow(tol / error, (1.0/4.0))
			print('delta=',delta,"",'error=',error/tol, "")
			
			#if (np.abs(delta*h) < np.abs(hmin) ):
		    	#	h = hmin #; error = 0.5*tol
			if it > 300:
				x = x + hmin ; y = y4
				break
					
			
		x = x + h
		y = y4
			
	else:
		if np.abs(delta*h) < np.abs(hmin):
			h = hmin
		elif np.abs(delta*h) > np.abs(hmax):
			h = hmax
		
		else:
			h = delta * h

		x = x + h
		y = y4"""

	if error < tol:
		if np.abs(delta*h) < np.abs(hmin):
			h = hmin
		elif np.abs(delta*h) > np.abs(hmax):
			h = hmax
		
		else:
			h = delta * h

		x = x + h
		y = y4
	   
	else:	
		it = 0
		while error > tol:
			h = delta * h ; it += 1


			K1 = h * F(x,y,*args) ; K2 = h * F(x + h/4, y + K1/4,*args) ; K3 = h * F(x + 3/8*h, y + 3/32*K1 + 9/32*K2,*args) ; K4 = h * F(x + 12/13*h, y + 1932/2197*K1 - 7200/2197*K2 + 7296/2197*K3, *args) ; K5 = h * F(x + h, y + 439/216*K1 - 8*K2 + 3680/513*K3 - 845/4104*K4, *args) ; K6 = h * F(x + h/2, y -8/27*K1 +2*K2 -3544/2565*K3 + 1859/4104*K4 -11/40*K5, *args) 

			y4 = y + 25/216*K1 + 1408/2565*K3 + 2197/4101*K4 -K5/5
			y5 = y + 16/135*K1 + 6656/12825*K3 + 28561/56430*K4 - 9/50*K5 + 2/55*K6
			
			error = np.linalg.norm(y5-y4)
			delta = pow(1.0/2.0,(1.0/4.0)) * pow(tol / error, (1.0/4.0))
			#print('delta=',delta,"",'error=',error/tol, "", it, h)
			
			if error < tol:
				if h < hmin:
					h = hmin
					x = x+h ; y = y4

					
				else:
					x = x + h ; y = y4

			#if (np.abs(delta*h) < np.abs(hmin) ):
		    	#	h = hmin #; error = 0.5*tol
			elif it > 500:
				if h < hmin :
					h = hmin ; x = x + h ; y = y4

					
		#x = x + h
		#y = y4
			
					
	return (h,x,y)  




@jit(nopython = True)
def geo(t,z,M,alfa,beta,rho0,z0,MD,b,hder):
    
    pphi = Pphi(rho0,z0,M,MD,b,alfa,beta)
    pt = Pt(rho0,z0,M,MD,b,alfa,beta)
    
    d2Rdt = -(0.5*math.exp(2*nu(z[0],z[2],M,MD,b,2)-lamb(z[0],z[2],M,MD,b,2))*derNU(z[0],z[2],M,MD,b,0,2,hder)* dt(z[0],z[2],M,MD,b,alfa,beta,pt)**2              +0.5*(dlamb2(z[0],z[2],M,MD,b,0,2,hder)-derNU(z[0],z[2],M,MD,b,0,2,hder))*z[1]**2 +               (dlamb2(z[0],z[2],M,MD,b,1,2,hder)-derNU(z[0],z[2],M,MD,b,1,2,hder))*z[1]*z[3]              -0.5*(dlamb2(z[0],z[2],M,MD,b,0,2,hder)-derNU(z[0],z[2],M,MD,b,0,2,hder))*z[3]**2              +0.5*math.exp(-lamb(z[0],z[2],M,MD,b,2))*z[0]*(-2+z[0]*derNU(z[0],z[2],M,MD,b,0,2,hder))*dphi(z[0],z[2],M,MD,b,alfa,beta,pphi)**2)
    
    d2Thedt = -(0.5*math.exp(2*nu(z[0],z[2],M,MD,b,2)-lamb(z[0],z[2],M,MD,b,2))*derNU(z[0],z[2],M,MD,b,1,2,hder)* dt(z[0],z[2],M,MD,b,alfa,beta,pt)**2                -0.5*(dlamb2(z[0],z[2],M,MD,b,1,2,hder)-derNU(z[0],z[2],M,MD,b,1,2,hder))*z[1]**2                + (dlamb2(z[0],z[2],M,MD,b,0,2,hder)-derNU(z[0],z[2],M,MD,b,0,2,hder))*z[1]*z[3]                + 0.5*(dlamb2(z[0],z[2],M,MD,b,1,2,hder)-derNU(z[0],z[2],M,MD,b,1,2,hder))*z[3]**2                + 0.5*math.exp(-lamb(z[0],z[2],M,MD,b,2))*z[0]**2*derNU(z[0],z[2],M,MD,b,1,2,hder)*dphi(z[0],z[2],M,MD,b,alfa,beta,pphi)**2)
    
    dPhidt = 1/gpp(z[0], z[2],M,MD,b)*pphi 



    return np.array([z[1], d2Rdt, z[3], d2Thedt,dPhidt])





@jit(nopython = True)
def func(y,x,h,alfa,beta,M,rho0,z0,MD,b,hder):
    
	it = 0
	#X=[]
	Y=[]
	#X.append(x)
	Y.append(y)
	#while (M + 0.5*(d2(y[0],y[2],M)+d1(y[0],y[2],M)) >= 2.01*M):# and y[0] > 0.0):
	#while (y[0] > 0.01):
	#if np.sqrt(y[0]**2+y[2]**2) > 30:
	while (nu(y[0],y[2],M,MD,b,2) > -3.0 and np.sqrt(gpp(np.sqrt(y[0]**2+y[2]**2),0,M,MD,b)) < 30.0):
		#start = time.time()
		#(y,x,h) = (  run_kut4_mod(geo, x, y, h,M,alfa,beta,rho0,z0,MD,b,hder)[2],run_kut4_mod(geo, x, y, h,M,alfa,beta,rho0,z0,MD,b,hder)[1],run_kut4_mod(geo, x, y, h,M,alfa,beta,rho0,z0,MD,b,hder)[0])
		(h,x,y) = run_kut4_mod(geo, x, y, h,M,alfa,beta,rho0,z0,MD,b,hder)
		#X.append(x)
		Y.append(y)
		it += 1
		#print( nu(y[0],y[2],M,MD,b,2),"",y[0],"",y[2],"",h, "",it) 

		#end = time.time()  
		     

		#if M + 0.5*(d2(y[0],y[2],M)+d1(y[0],y[2],M)) > 30:
		if np.sqrt(gpp(np.sqrt(y[0]**2+y[2]**2),0,M,MD,b)) >= 30.0:
			yf = [Y[-1][0],Y[-1][2],Y[-1][4]]
			break

		####   Pontos que caem no buraco negro    #####

		elif nu(y[0],y[2],M,MD,b,2) <= -3.0:
        #elif np.abs(gtt(y[0],y[2],M,MD,b)) < 10**-2  
	    		yf = [0.001,0.001,0.2]
	    		break
		

		#elif ( (M + 0.5*(d2(y[0],y[2],M)+d1(y[0],y[2],M))) >= 6.0*M and np.sign(Y[-1][2]) == - np.sign(Y[-2][2]) ):
		#elif ( y[0] > b and np.sign(Y[-1][2]) == - np.sign(Y[-2][2]) ):
		#	yf = [Y[-1][0],Y[-1][2]+50.0]
                #yf = [Y[-1][0],Y[-1][2]]
		#	break
		
		#elif ( gzz(y[0],y[2],M,MD,b) < 10**-10 and grr(y[0],y[2],M,MD,b) < 10**-10):
		#elif ( np.abs(dr(y[0],y[2],M,MD,b,alfa,beta)) > 20.0 and  np.abs(dthe(y[0],y[2],M,MD,b,alfa)) > 20.0):
		#	yf = [Y[-1][0]+50.0,Y[-1][2]]
		#	break
		    



	#Ynew0 = [item[0] for item in Y]
	#Ynew1 = [item[1] for item in Y]
	#Ynew2 = [item[2] for item in Y]
	#Ynew3 = [item[3] for item in Y]
	#return(X,Ynew0,Ynew1,Ynew2,Ynew3)
	return (yf)




start = time.time()
M = 0.9
MD = 0.1
z0 = 0.0
b = 6.0
hder = 10**-6
func_initial = lambda R : np.sqrt(gpp_i(R,z0,M,MD,b)) - 15.0

R_initial_guess = 10.0
R_solution = fsolve(func_initial, R_initial_guess)

rho0 = float(R_solution) 
print(rho0)



hder = 10**-6

alfaa = np.linspace(-np.arctan(10/15),np.arctan(10/15),1000)
betaa = np.linspace(np.arctan(10/15),-np.arctan(10/15),1000)

alfa = np.linspace(alfaa[0],alfaa[int(len(alfaa)/2)-1],int(len(alfaa)/2))
beta = np.linspace(betaa[0],betaa[int(len(betaa)/2)-1],int(len(betaa)/2))



Mat = np.zeros((len(alfa),len(beta)))
Mz = np.zeros((len(alfa),len(beta)))
Mphi = np.zeros((len(alfa),len(beta)))

for i in range(len(alfa)):
	for j in range(len(beta)):

		
		#start1 = time.time()
		y = np.array([rho0,dr(rho0,z0,M,MD,b,alfa[i],beta[j]), z0,dthe(rho0,z0,M,MD,b,alfa[i]),0])
		(Mat[i,j],Mz[i,j],Mphi[i,j]) = func(y,300.0,-0.02,alfa[i],beta[j],M,rho0,z0,MD,b,hder)

		#end1 = time.time()
		
		#print(end1-start1,alfa[i],beta[j])

end = time.time()
print(end-start)


np.savetxt('Mat',Mat)
np.savetxt('Mz',Mz)
np.savetxt('Mphi',Mphi)



#############################################################
#############################################################

########            UNCOMMENT           ####################

#############################################################
#############################################################
#############################################################

"""
Mat2 = np.zeros((2*len(Mat),2*len(Mat[0])))
Mz2 = np.zeros((2*len(Mz),2*len(Mz[0])))
Mphi2 = np.zeros((len(Mphi),len(Mphi[0])))
Mphi3 = np.zeros((2*len(Mphi2),2*len(Mphi2[0])))
M2 = np.zeros((len(Mat2),len(Mat2[0])))


for i in range(len(Mphi)):
	for j in range(len(Mphi[0])):
        	if Mphi[i,j] > 2*np.pi:
            		Mphi2[i,j] = Mphi[i,j]-int(Mphi[i,j]/(2*np.pi))*2*np.pi
            
        	elif (Mphi[i,j] < 0 and Mphi[i,j] > -2*np.pi):
            		Mphi2[i,j] = Mphi[i,j] + 2*np.pi
            
        	elif Mphi[i,j] < -2*np.pi:
            		Mphi2[i,j] = Mphi[i,j] - int(Mphi[i,j]/(2*np.pi))*2*np.pi + 2*np.pi
            
        	else:
            		Mphi2[i,j] = Mphi[i,j]


########    RADIAL COORDINATE   ############

for i in range(len(Mat)):
	for j in range(len(Mat[0])):
		Mat2[i,j] = Mat[i,j]
		Mat2[i,-j-1] = Mat[i,j]
		Mat2[-i-1,j] = Mat[i,j]
		Mat2[-i-1,-j-1] = Mat[i,j]


########    PHI COORDINATE   ############

for i in range(len(Mphi2)):
	for j in range(len(Mphi2[0])):
		Mphi3[i,j]       = Mphi2[i,j]
		Mphi3[i,-j-1]    = 2*np.pi - Mphi2[i,j]
		Mphi3[-i-1,j]    = Mphi2[i,j]
		Mphi3[-i-1,-j-1] = 2*np.pi - Mphi2[i,j]

########    Z COORDINATE   ############

for i in range(len(Mz)):
	for j in range(len(Mz[0])):
		Mz2[i,j] = Mz[i,j]
		Mz2[i,-j-1] = Mz[i,j]
		Mz2[-i-1,j] = - Mz[i,j]
		Mz2[-i-1,-j-1] = -Mz[i,j]


for i in range(len(Mat2)):
	for j in range(len(Mat2[0])):
        	if Mat2[i,j] < 0.002:
            		M2[i,j] = 0
        	else:
            		if (Mz2[i,j] < 0 and (Mphi3[i,j] >= 0 and Mphi3[i,j] <= np.pi)):
                		M2[i,j] = 3 #1
                
            		elif ( Mz2[i,j] < 0 and (Mphi3[i,j] > np.pi and Mphi3[i,j] <= 2*np.pi)):   
                		M2[i,j] = 4 #2
                
            		elif (Mz2[i,j] >= 0 and (Mphi3[i,j] > 0  and Mphi3[i,j] <= np.pi)):
                		M2[i,j] = 2 #3
                
            		elif ( Mz2[i,j] >= 0 and (Mphi3[i,j] > np.pi and Mphi3[i,j] <= 2*np.pi )):
                		M2[i,j] = 1 #4
            
                


plt.figure(figsize = (25,25))

#c_map = [[0,0,0],[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0]]#,[1,1,1]]#,[0,1,1]]
c_map = [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0]]

cm = matplotlib.colors.ListedColormap(c_map)

#plt.imshow(M2, cmap = cm,extent=[-np.sqrt(gpp(r0,a,M,theta))*beta[0],-np.sqrt(gpp(r0,a,M,theta))*beta[-1],np.sqrt(gpp(r0,a,M,theta))*alfa[0],np.sqrt(gpp(r0,a,M,theta))*alfa[-1]])
#plt.imshow(M22, cmap = "binary",extent=[beta[0],beta[-1],alfa[0],alfa[-1]])

plt.imshow(M2, cmap = cm,extent=[-beta[0],-beta[-1],alfa[0],alfa[-1]])



plt.xlabel("x(M)",size=40)
plt.ylabel("y(M)",size=40)

plt.tick_params(axis='both', which='major', labelsize=30)

#plt.show()
plt.savefig("Schwarzschild_WeylCoords_lensing_154x154")

#############################################################
#############################################################
#############################################################
#############################################################
#############################################################


"""

#####################		( R , THETA ) 		###################################
"""
Mphi2 = np.zeros((len(alfa),len(beta)))
Mthet2 = np.zeros((len(alfa),len(beta)))
Mat2 = np.zeros((len(alfa),len(beta)))

for i in range(len(alfa)):
	for j in range(len(beta)):
        	Mat2[i,j] = 1+0.5*(d1(Mat[i,j],Mz[i,j],1)+d2(Mat[i,j],Mz[i,j],1))

for i in range(len(alfa)):
	for j in range(len(beta)):
        #Mthet2[i,j] = np.arccos(Mz[i,j]/(np.sqrt(Mat[i,j]**2+Mz[i,j]**2)-1)) 
        	Mthet2[i,j] = np.arccos(0.5*(np.sqrt(Mat[i,j]**2+(Mz[i,j]+1)**2)-np.sqrt(Mat[i,j]**2+(Mz[i,j]-1)**2)))
        
for i in range(len(alfa)):
	for j in range(len(beta)):
        	if Mphi[i,j] > 2*np.pi:
            		Mphi2[i,j] = Mphi[i,j]-int(Mphi[i,j]/(2*np.pi))*2*np.pi
            
        	elif (Mphi[i,j] < 0 and Mphi[i,j] > -2*np.pi):
            		Mphi2[i,j] = Mphi[i,j] + 2*np.pi
            
        	elif Mphi[i,j] < -2*np.pi:
            		Mphi2[i,j] = Mphi[i,j] - int(Mphi[i,j]/(2*np.pi))*2*np.pi + 2*np.pi
            
        	else:
            		Mphi2[i,j] = Mphi[i,j]
            

dang = 0.01
kmax_the = np.pi/ (10*np.pi/180)
kmax_phi = 2*np.pi/ (10*np.pi/180)
            
M2 = np.zeros((len(Mat),len(Mat[0])))
for i in range(len(Mat)):
	for j in range(len(Mat[0])):
        #if (Mat[i,j] < 0.08 and np.abs(Mz[i,j]) < 1.01) :
        	if Mat2[i,j] < 2.2:
            		M2[i,j] = 0
        	else:
            		if ((Mthet2[i,j] >= np.pi/2 and Mthet2[i,j] <= np.pi) and (Mphi2[i,j] >= 0 and Mphi2[i,j] <= np.pi)):
                		M2[i,j] = 3 #1
                
            		elif ((Mthet2[i,j] >= np.pi/2  and Mthet2[i,j] <= np.pi) and (Mphi2[i,j] > np.pi and Mphi2[i,j] <= 2*np.pi)):   
                		M2[i,j] = 4 #2
                
            		elif ((Mthet2[i,j] >= 0 and Mthet2[i,j] < np.pi/2) and (Mphi2[i,j] > 0  and Mphi2[i,j] <= np.pi)):
                		M2[i,j] = 2 #3
                
            		elif ((Mthet2[i,j] >= 0 and Mthet2[i,j] <= np.pi/2) and (Mphi2[i,j] > np.pi and Mphi2[i,j] <= 2*np.pi )):
                		M2[i,j] = 1 #4
            
                
for i in range(len(Mat)):
	for j in range(len(Mat[0])):
        	"Einstein ring"
		#if (( ((Mthet2[i,j] < np.pi/2 + 10*np.pi/180 and Mthet2[i,j]> np.pi/2-10*np.pi/180) and \
		#     (Mphi2[i,j]< np.pi + 10*np.pi/180 and Mphi2[i,j]> np.pi -10*np.pi/180)) and \
		#   ((Mphi2[i,j]-np.pi)**2 + (Mthet2[i,j]-np.pi/2)**2 <= np.pi * (np.pi*10/180)**2) ) and \
		#   Mat[i,j] >=0.02):
		#        M2[i,j] = 5  

        	"Black lines of the grid"
        	kmax = kmax_the
        	for k in range(0,int(kmax)+1):
            		if (Mthet2[i,j] <= k*10*np.pi/180 + dang and Mthet2[i,j] >= k*10*np.pi/180-dang):
                		M2[i,j] = 0
        
        	kmax = kmax_phi
        	for k in range(0,int(kmax)+1):
            		if (Mphi2[i,j] <= k*10*np.pi/180 + dang and Mphi2[i,j] >= k*10*np.pi/180-dang):
                		M2[i,j] = 0
                     
             
            

plt.figure(figsize = (25,25))

#c_map = [[0,0,0],[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0]]#,[1,1,1]]#,[0,1,1]]
c_map = [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0]]

cm = matplotlib.colors.ListedColormap(c_map)

#plt.imshow(M2, cmap = cm,extent=[-np.sqrt(gpp(r0,a,M,theta))*beta[0],-np.sqrt(gpp(r0,a,M,theta))*beta[-1],np.sqrt(gpp(r0,a,M,theta))*alfa[0],np.sqrt(gpp(r0,a,M,theta))*alfa[-1]])
#plt.imshow(M22, cmap = "binary",extent=[beta[0],beta[-1],alfa[0],alfa[-1]])

plt.imshow(M2, cmap = cm,extent=[-beta[0],-beta[-1],alfa[0],alfa[-1]])

#plt.plot(x(r,a,M,theta,r0),yplus(r,a,M,theta,r0),'o',color="red", label="analytical")
#plt.plot(x(r,a,M,theta,r0),yminus(r,a,M,theta,r0),'o',color="red",label = "analytical")

plt.xlabel("x(M)",size=40)
plt.ylabel("y(M)",size=40)

plt.tick_params(axis='both', which='major', labelsize=30)
#plt.legend(loc=10, bbox_to_anchor=(0.85, 0.9), ncol=1,fontsize=17)
#plt.xlim(-np.arctan(10/15),np.arctan(10/15))
#plt.ylim(-np.arctan(10/15),np.arctan(10/15))
#plt.show()
plt.savefig("Schwarzschild_WeylCoords_lensing_154x154")
"""


