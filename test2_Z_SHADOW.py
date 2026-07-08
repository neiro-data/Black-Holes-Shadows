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


#########             Inverted MORGAN MORGAN DISCS         ###############


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



################## SCHWARZSCHILD BH + DISK case ###################     

# l = 0 corresponds to derivative in respect to RHO
# l = 1 corresponds to derivative in respect to Z
# m = 0 corresponds to nuSch
# m = 1 corresponds to nuD

##################################################################### 


@jit(nopython = True)
def nu(R,z,M,MD,a): #Schwarzschild \nu potential: nuSch
    
    #BH
	nuSch = math.log((d1(R,z,M) + d2(R,z,M) - 2*M)/(d1(R,z,M) + d2(R,z,M) +2*M))

    #DISK
	if xi2(R,z,a) == 0.0:
        	nuD = ( (MD*np.sqrt(np.abs(a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))*(-3*a**2 + R**2 + z**2 + 3*np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))/(np.sqrt(2)*np.pi*(R**2 + z**2)**2) - (2*MD*(-(a**2*(R**2 - 2*z**2)) + 2*(R**2 + z**2)**2)*np.pi/2)/(np.pi*(R**2 + z**2)**2.5) )
	else:
		nuD = ( (MD*np.sqrt(np.abs(a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))*(-3*a**2 + R**2 + z**2 + 3*np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))/(np.sqrt(2)*np.pi*(R**2 + z**2)**2) - (2*MD*(-(a**2*(R**2 - 2*z**2)) + 2*(R**2 + z**2)**2)*np.arctan(np.sqrt(2)*np.sqrt((R**2 + z**2)/np.abs((a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4))))))/(np.pi*(R**2 + z**2)**2.5) )
		
	return (nuSch + nuD)


@jit(nopython = True)
def lambSch(rho,z,M,MD,b): #Schwarzshild \lambda potential: lambSch
    sigma = np.sqrt((rho**2 + z**2 + M**2)**2 - 4*z**2*M**2)
    return np.log( ((d1(rho,z,M) + d2(rho,z,M))**2-4*M**2)/(4*sigma) )


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
def derNU(rho,z,M,MD,b,l,hder):
    return derivative(nu,l,rho,z,hder,M,MD,b)
    

@jit(nopython = True)
def dlamb(z,rho,M,MD,b,l,hder):
    if l == 0:
        return 0.5*rho * ( derNU(rho,z,M,MD,b,0,hder)**2 - derNU(rho,z,M,MD,b,1,hder)**2)
    
    elif l == 1:
        return rho * derNU(rho,z,M,MD,b,0,hder) * derNU(rho,z,M,MD,b,1,hder)

@jit(nopython = True)
def dlamb2(rho,z,M,MD,b,l,hder):
    if l == 0:
        return 0.5*rho * ( derNU(rho,z,M,MD,b,0,hder)**2 - derNU(rho,z,M,MD,b,1,hder)**2)
    
    elif l == 1:
        return rho * derNU(rho,z,M,MD,b,0,hder) * derNU(rho,z,M,MD,b,1,hder)


"""@jit
def lamb_Mat(rho,z,M,MD,b,m,hder):
    limit = 0.0
    if z >= limit:
        return sci.quad(dlamb,40.0,z,(rho,M,MD,b,1,m,hder))[0] + sci.quad(dlamb2,40.0,rho,(40.0,M,MD,b,0,m,hder))[0] + lambSch(40.0,40.0,M+MD,MD,b)
    
    elif z < -limit:
        return sci.quad(dlamb,-40,z,(rho,M,MD,b,1,m,hder))[0] + sci.quad(dlamb2,40.0,rho,(-40.0,M,MD,b,0,m,hder))[0] + lambSch(40.0,-40.0,M+MD,MD,b)

"""
@jit(nopython = True)    
def lamb(rho,z,M,MD,b):
 
	zref = 40 ; Rref = 40

	iR , jR =  (zref-z)/zref * (len(Mat_nu)-1)/2 , (Rref-rho)/(Rref-0.0)*(len(Mat_nu[0])-1) 
	

	i0 , j0 =  int( math.floor( (zref-z)/zref * (len(Mat_nu)-1)/2 ) ) , int( math.floor( (Rref-rho)/(Rref-0.0)*(len(Mat_nu[0])-1) ) ) 


	if np.abs(jR) > len(Mat_nu[0])-1:
		iR,jR =  int(len(Mat_nu)/2)+0.5 , len(Mat_nu[0])-0.5 
		i0,j0 =  int(math.floor(iR)), int(math.floor(jR)) 

	I,J = iR - i0 , jR - j0

	#print(iR,"",jR,"",i0,"",j0)

	f00,f01,f10,f11 =  Mat_nu[i0 , j0] , Mat_nu[i0 , j0+1] , Mat_nu[i0+1 , j0] ,Mat_nu[i0+1 , j0+1] 

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



Mat_nu = np.loadtxt("Mat_nu_disk0.9")


# In[32]:




@jit(nopython = True)
def gtt(rho,z,M,MD,b):
    return -math.exp(nu(rho,z,M,MD,b))



@jit(nopython = True)
def grr(rho,z,M,MD,b):
    return math.exp( lamb(rho,z,M,MD,b) - nu(rho,z,M,MD,b) ) 

@jit(nopython = True)
def gzz(rho,z,M,MD,b):
    return math.exp( lamb(rho,z,M,MD,b) - nu(rho,z,M,MD,b) ) 

@jit(nopython = True)
def gpp(rho,z,M,MD,b):
    return rho**2 * math.exp(- nu(rho,z,M,MD,b))

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
def Pt(rho,z,M,MD,b):
    return -1/zeta(rho,z,M,MD,b)

@jit(nopython = True)
def dphi(rho,z,M,MD,b,p_phi):
    return 1/gpp(rho,z,M,MD,b)*p_phi

@jit(nopython = True)
def dt(rho,z,M,MD,b,p_t):
    return 1/gtt(rho,z,M,MD,b)*p_t


# In[33]:


@jit(nopython = True)
def run_kut4_mod(F,x,y,h,*args):
    
	hmax = -0.04
	hmin = -10**-7	
	tol = 10**-3
	
	K1 = h * F(x,y,*args)
	K2 = h * F(x + h/4, y + K1/4,*args)
	K3 = h * F(x + 3/8*h, y + 3/32*K1 + 9/32*K2,*args)
	K4 = h * F(x + 12/13*h, y + 1932/2197*K1 - 7200/2197*K2 + 7296/2197*K3, *args)
	K5 = h * F(x + h, y + 439/216*K1 - 8*K2 + 3680/513*K3 - 845/4104*K4, *args)
	K6 = h * F(x + h/2, y -8/27*K1 +2*K2 -3544/2565*K3 + 1859/4104*K4 -11/40*K5, *args) 

	y4,y5 =  y + 25/216*K1 + 1408/2565*K3 + 2197/4101*K4 -K5/5 , y + 16/135*K1 + 6656/12825*K3 + 28561/56430*K4 - 9/50*K5 + 2/55*K6
	


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


			K1 = h * F(x,y,*args)
			K2 = h * F(x + h/4, y + K1/4,*args)
			K3 = h * F(x + 3/8*h, y + 3/32*K1 + 9/32*K2,*args)
			K4 = h * F(x + 12/13*h, y + 1932/2197*K1 - 7200/2197*K2 + 7296/2197*K3, *args)
			K5 = h * F(x + h, y + 439/216*K1 - 8*K2 + 3680/513*K3 - 845/4104*K4, *args)
			K6 = h * F(x + h/2, y -8/27*K1 +2*K2 -3544/2565*K3 + 1859/4104*K4 -11/40*K5, *args) 

			y4,y5 =  y + 25/216*K1 + 1408/2565*K3 + 2197/4101*K4 -K5/5 , y + 16/135*K1 + 6656/12825*K3 + 28561/56430*K4 - 9/50*K5 + 2/55*K6
			
			
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
	pt = Pt(rho0,z0,M,MD,b)
	
	d2Rdt,d2Thedt =     -(0.5*math.exp(2*nu(z[0],z[2],M,MD,b)-lamb(z[0],z[2],M,MD,b))*derNU(z[0],z[2],M,MD,b,0,hder)* dt(z[0],z[2],M,MD,b,pt)**2              +0.5*(dlamb2(z[0],z[2],M,MD,b,0,hder)-derNU(z[0],z[2],M,MD,b,0,hder))*z[1]**2 +               (dlamb2(z[0],z[2],M,MD,b,1,hder)-derNU(z[0],z[2],M,MD,b,1,hder))*z[1]*z[3]              -0.5*(dlamb2(z[0],z[2],M,MD,b,0,hder)-derNU(z[0],z[2],M,MD,b,0,hder))*z[3]**2              +0.5*math.exp(-lamb(z[0],z[2],M,MD,b))*z[0]*(-2+z[0]*derNU(z[0],z[2],M,MD,b,0,hder))*dphi(z[0],z[2],M,MD,b,pphi)**2),     -(0.5*math.exp(2*nu(z[0],z[2],M,MD,b)-lamb(z[0],z[2],M,MD,b))*derNU(z[0],z[2],M,MD,b,1,hder)* dt(z[0],z[2],M,MD,b,pt)**2                -0.5*(dlamb2(z[0],z[2],M,MD,b,1,hder)-derNU(z[0],z[2],M,MD,b,1,hder))*z[1]**2                + (dlamb2(z[0],z[2],M,MD,b,0,hder)-derNU(z[0],z[2],M,MD,b,0,hder))*z[1]*z[3]                + 0.5*(dlamb2(z[0],z[2],M,MD,b,1,hder)-derNU(z[0],z[2],M,MD,b,1,hder))*z[3]**2                + 0.5*math.exp(-lamb(z[0],z[2],M,MD,b))*z[0]**2*derNU(z[0],z[2],M,MD,b,1,hder)*dphi(z[0],z[2],M,MD,b,pphi)**2)    
    



	return np.array([z[1], d2Rdt, z[3], d2Thedt])


@jit(nopython = True)
def func(y,x,h,alfa,beta,M,rho0,z0,MD,b,hder):
    
	#it = 0
	#X=[]
	#Y=[]
	#X.append(x)
	#Y.append(y)

	###	TENTATIVA	###
	#X = np.zeros(1)
	Y = np.zeros((1,4))
	
	#X = np.concatenate((X , x))
	Y = np.concatenate((Y , y.reshape((1,4)) ) , axis = 0)

	###	TENTATIVA2	###
	


	#while (M + 0.5*(d2(y[0],y[2],M)+d1(y[0],y[2],M)) >= 2.01*M):# and y[0] > 0.0):
	while (nu(y[0],y[2],M,MD,b) > -3.0 and np.sqrt(gpp(np.sqrt(y[0]**2+y[2]**2),0,M,MD,b)) < 30.0):

		#(y,x,h) = (  run_kut4_mod(geo, x, y, h,M,alfa,beta,rho0,z0,MD,b,hder)[2],run_kut4_mod(geo, x, y, h,M,alfa,beta,rho0,z0,MD,b,hder)[1],run_kut4_mod(geo, x, y, h,M,alfa,beta,rho0,z0,MD,b,hder)[0])
		(h,x,y) = run_kut4_mod(geo, x, y, h,M,alfa,beta,rho0,z0,MD,b,hder)
		#X.append(x)
		#Y.append(y)

		### 	 TENTATIVA 	###
		#X = np.concatenate((X , x))
		Y = np.concatenate( (Y , y.reshape((1,4)) ) , axis = 0)
		#it += 1
		#print( nu(y[0],y[2],M,MD,b,2),"",y[0],"",y[2],"",h, "",it) 

		     

		#if M + 0.5*(d2(y[0],y[2],M)+d1(y[0],y[2],M)) > 30:
		if np.sqrt(gpp(np.sqrt(y[0]**2+y[2]**2),0,M,MD,b)) >= 30.0:
			yf = [Y[-1][0],Y[-1][2]]
			#yf = [y[0],y[2]]
			break

		####   Pontos que caem no buraco negro    #####

		elif nu(y[0],y[2],M,MD,b) <= -3.0:
        #elif np.abs(gtt(y[0],y[2],M,MD,b)) < 10**-2  
	    		yf = [0.001,0.001]
	    		break
		
        	####   Pontos do Disco    #####

		#elif ( (M + 0.5*(d2(y[0],y[2],M)+d1(y[0],y[2],M))) >= 6.0*M and np.sign(Y[-1][2]) == - np.sign(Y[-2][2]) ):
		elif ( y[0] > b and np.sign(Y[-1][2]) == - np.sign(Y[-2][2]) ):
			yf = [Y[-1][0],Y[-1][2]+50.0]

			break
		
		
		



	#Ynew0 = [item[0] for item in Y]
	#Ynew1 = [item[1] for item in Y]
	#Ynew2 = [item[2] for item in Y]
	#Ynew3 = [item[3] for item in Y]
	#return(X,Ynew0,Ynew1,Ynew2,Ynew3)
	return (yf)




start = time.time()
M = 0.1
MD = 0.9
z0 = 0.0
b = 6.0
hder = 10**-6
func_initial = lambda R : np.sqrt(gpp_i(R,z0,M,MD,b)) - 15.0

R_initial_guess = 10.0
R_solution = fsolve(func_initial, R_initial_guess)

rho0 = float(R_solution) 
print(rho0)




alfaa = np.linspace(-np.arctan(10/15),np.arctan(10/15),88)
betaa = np.linspace(np.arctan(10/15),-np.arctan(10/15),88)



alfa = np.linspace(alfaa[0],alfaa[int(len(alfaa)/2)-1],int(len(alfaa)/2))
beta = np.linspace(betaa[0],betaa[int(len(betaa)/2)-1],int(len(betaa)/2))


Mat = np.zeros((len(alfa),len(beta)))
Mz = np.zeros((len(alfa),len(beta)))
Mphi = np.zeros((len(alfa),len(beta)))

for i in range(len(alfa)):
	for j in range(len(beta)):

		start1 = time.time()
		y = np.array([rho0,dr(rho0,z0,M,MD,b,alfa[i],beta[j]), z0,dthe(rho0,z0,M,MD,b,alfa[i])])

		(Mat[i,j],Mz[i,j]) = func(y,300.0,-0.02,alfa[i],beta[j],M,rho0,z0,MD,b,hder)

		end1 = time.time()
		
		print(end1-start1,alfa[i],beta[j])

end = time.time()
print(end-start)


np.savetxt('Mat',Mat)
np.savetxt('Mz',Mz)






M2 = np.zeros((len(Mat),len(Mat[0])))
Mat2 = np.zeros((len(alfa),len(beta)))
Mz2 = np.zeros((len(alfa),len(beta)))




"""
for i in range(len(Mz)):
	for j in range(len(Mz[0])):
		if Mz[i,j] > 49.0:
			Mz2[i,j] = Mz[i,j]-50.0  
		else:
			Mz2[i,j] = Mz[i,j]           

for i in range(len(alfa)):
	for j in range(len(beta)):
		Mat2[i,j] = M+0.5*(d1(Mat[i,j],Mz2[i,j],M)+d2(Mat[i,j],Mz2[i,j],M))
count = 0
for i in range(len(Mat)):
	for j in range(len(Mat[0])):

		if Mat2[i,j] <= 2.01*M:
    			M2[i,j] = 1
    			count +=1

		elif (Mat2[i,j] > 6.0*M and Mz[i,j] > 49.0):
			M2[i,j] = 2        	
		
		else:
    			M2[i,j] = 0
print(count) 
  
"""


for i in range(len(Mz)):
	for j in range(len(Mz[0])):
		if Mz[i,j] > 49.0:
			Mz2[i,j] = Mz[i,j]-50.0  
		else:
			Mz2[i,j] = Mz[i,j]           

for i in range(len(Mat)):
	for j in range(len(Mat[0])):
		if Mat[i,j] > 50.0:
			Mat2[i,j] = Mat[i,j] -50.0
		else:
			Mat2[i,j] = Mat[i,j]
count1 = 0
count2 = 0
for i in range(len(Mat)):
	for j in range(len(Mat[0])):

		if (Mat2[i,j] <= 0.002):# and Mz[i,j] < M):
    			M2[i,j] = 1
    			count1 +=1

		elif (Mat2[i,j] > b and Mz[i,j] > 49.0):
			M2[i,j] = 2    
	  	
		
		else:
    			M2[i,j] = 0
print(count1,count2)   


plt.figure(figsize = (15,15))


#c_map = [[1,1,1],[0, 0, 0], [0.192, 0.192, 0.192],[1,0,0]]
c_map = [[1,1,1],[0, 0, 0], [0.192, 0.192, 0.192]]
cm = matplotlib.colors.ListedColormap(c_map)

plt.imshow(M2, cmap = cm,extent=[-beta[0],-beta[-1],alfa[0],alfa[-1]])
#plt.colorbar()


plt.xlabel("$\\beta$",size=30)
plt.ylabel("$\\alpha$",size=30)

plt.tick_params(axis='both', which='major', labelsize=14)
#plt.legend(loc=10, bbox_to_anchor=(0.85, 0.9), ncol=1,fontsize=17)

#plt.show()
plt.savefig("Schwarzschild_Weyl_coords_154x154")            




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


"""


start = time.time()
M = 0.9
MD = 0.1
rho = 15.0
z0 = 0.0
b = 6.0
rho0 = np.sqrt(gpp(rho,z0,M,MD,b))
hder = 10**-6


#alfa = -0.3311738801589748 
#beta = 0.006758650615489303
alfa = -0.2
beta = 0.006758650615489303

y = np.array([rho0,dr(rho0,z0,M,MD,b,alfa,beta), z0,dthe(rho0,z0,M,MD,b,alfa)])

X = (func(y,300.0,-0.02,alfa,beta,M,rho0,z0,MD,b,hder)[0])
Y1 = (func(y,300.0,-0.02,alfa,beta,M,rho0,z0,MD,b,hder)[1])
Y2 = (func(y,300.0,-0.02,alfa,beta,M,rho0,z0,MD,b,hder)[2])
Y3 = (func(y,300.0,-0.02,alfa,beta,M,rho0,z0,MD,b,hder)[3])
Y4 = (func(y,300.0,-0.02,alfa,beta,M,rho0,z0,MD,b,hder)[4])

end = time.time()
print(end-start)


# In[24]:

print("rho_min=",np.min(Y1))
print("r_min=",np.min( M+0.5*(d1(np.transpose(Y1),np.transpose(Y3),M)+d2(np.transpose(Y1),np.transpose(Y3),M)) ))
plt.figure(figsize=(20,20))
NU = np.zeros(len(X))
DRHO = np.zeros(len(X))
DZ = np.zeros(len(X))
dNU = np.zeros(len(X))
for i in range(len(X)):
#    dNU[i] = derNU(np.transpose(Y1)[i],np.transpose(Y3)[i],1.0,1.0,6.0,1,2,10**-6)
	NU[i] = nu(np.transpose(Y1)[i],np.transpose(Y3)[i],M,MD,b,0)
	DRHO[i] = dr(np.transpose(Y1)[i],np.transpose(Y3)[i],M,MD,b,alfa,beta)
	DZ[i] = dthe(np.transpose(Y1)[i],np.transpose(Y3)[i],M,MD,b,alfa)
print(np.min(NU))
plt.plot(np.transpose(X),NU,'.',label="nu")
plt.plot(np.transpose(X),DRHO,'.',label="DRHO")
plt.plot(np.transpose(X),DZ,'.', label="DZ")
#plt.plot(np.transpose(X),dNU,'.',label="dNU")
#plt.plot(np.transpose(X),derNU(np.transpose(Y3),np.transpose(Y1),1,1,1,1,1,10**-8),'.',label="dNUz")
#plt.plot(np.transpose(X),dlambr(np.transpose(Y1),np.transpose(Y3),1,1,1,1,1,10**-8),'.',label="dlambr")
#plt.plot(np.transpose(X),dlambz(np.transpose(Y3),np.transpose(Y1),1,1,1,1,1,10**-8),'.',label="dlambz")
#plt.plot(np.transpose(X),nu(np.transpose(Y1),np.transpose(Y3),1,1,1,1,1),'.',label="NU")
#plt.plot(np.transpose(X),lamb(np.transpose(Y1),np.transpose(Y3),1,1,1,1,1),'.',label="LAMB")

   

plt.plot(np.transpose(X),np.transpose(Y1),'.',label= "rho")
#plt.plot(np.transpose(X),np.transpose(Y2),'.',label="drho")
plt.plot(np.transpose(X),np.transpose(Y3),'.',label = "z")
#plt.plot(np.transpose(X),np.transpose(Y4),'.',label="dz")

#plt.axhline(M*2.0)
plt.axhline(0.0)
plt.plot(np.transpose(X),M+0.5*(d1(np.transpose(Y1),np.transpose(Y3),M)+d2(np.transpose(Y1),np.transpose(Y3),M)),'.',label="r")

plt.legend(loc=10, bbox_to_anchor=(0.9, 0.8), ncol=2,fontsize=20)
plt.xlabel("lambda = proper time",size = 16)
plt.ylabel("$\\rho$", size = 16)
plt.ylim(-30,30)
#plt.axhline(0.0, color = "red")
#plt.axhline(1.0, color = "green")
#plt.axhline(2.0)
#plt.xlim(280,300)
#plt.ylim(-2,3)
plt.show()


"""

