#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.integrate as sci
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.integrate import ode
import matplotlib.pyplot as plt
import matplotlib.colors
import time
from mpl_toolkits.mplot3d import Axes3D
from numba import jit



# In[2]:


#Set |P| = 1
#METRIC ELEMENTS
@jit(nopython = True)
def gtt(r,a,M,theta):
    return(-(1-2*M*r/(r**2 + (a*M)**2 * np.cos(theta)**2))) 

@jit(nopython = True)
def gtp(r,a,M,theta):
    return (-2*M*(a*M)*r*(np.sin(theta))**2/(r**2 + (a*M)**2 * np.cos(theta)**2))

@jit(nopython = True)
def gpp(r,a,M,theta):
    return ((np.sin(theta))**2*(r**2 + (a*M)**2 + 2*M*r*(a*M)**2*(np.sin(theta))**2/(r**2 + (a*M)**2 * np.cos(theta)**2)))

@jit(nopython = True)
def grr(r,a,M,theta):
    return ((r**2 + (a*M)**2*np.cos(theta)**2)/(r**2-2*M*r+(a*M)**2))

@jit(nopython = True)
def gthe(r,a,M,theta):
    return ((r**2+(a*M)**2*np.cos(theta)**2))

@jit(nopython = True)
def gpp_inv(r,a,M,theta):
    return ((((a*M)**2 + 2*r*(r-2*M) + (a*M)**2*np.cos(2*theta))*(1/np.sin(theta)**2))            /(((a*M)**2 +r*(r-2*M))*((a*M)**2 + 2*r**2 + (a*M)**2*np.cos(2*theta))))

@jit(nopython = True)
def gtt_inv(r,a,M,theta):
    return (-((a*M)**4 + 2*r**4 + (a*M)**2*r*(2*M+3*r)+(a*M)**2*((a*M)**2 + r*(r-2*M))*np.cos(2*theta))              /(((a*M)**2 + r*(r-2*M))*((a*M)**2 + 2*r**2 + (a*M)**2*np.cos(2*theta))))

@jit(nopython = True)
def gtp_inv(r,a,M,theta):
    return (-4*(a*M)*M*r/(((a*M)**2 + r*(r-2*M))*((a*M)**2 + 2*r**2 + (a*M)**2*np.cos(2*theta))))

#------------------------\\----------------------
#AUXILIAR FUNCTIONS
@jit(nopython = True)
def gamma (r,a,M,theta):
    return ((-gtp(r,a,M,theta)/gpp(r,a,M,theta))*np.sqrt(gpp(r,a,M,theta)            /(gtp(r,a,M,theta)**2-gtt(r,a,M,theta)*gpp(r,a,M,theta))))

@jit(nopython = True)
def zeta(r,a,M,theta):
    return (np.sqrt(gpp(r,a,M,theta)/(gtp(r,a,M,theta)**2-gtt(r,a,M,theta)*gpp(r,a,M,theta))))

@jit(nopython = True)
def delta(r,a,M):
    return (r**2 -2*M*r +(a*M)**2)
#------------------------\\----------------------
#COORIDNATES DERIVATIVES
@jit(nopython = True)
def dthe(r,a,M,theta,alfa):
    return (np.sqrt(gthe(r,a,M,theta))*np.sin(alfa)/             (r**2+(a*M)**2*np.cos(theta)**2))

@jit(nopython = True)
def dr(r,a,M,theta,alfa,beta):
    return (np.sqrt(grr(r,a,M,theta))*np.cos(beta)*np.cos(alfa)*            (r**2-2*M*r+(a*M)**2)/(r**2+(a*M)**2*np.cos(theta)**2))

@jit(nopython = True)
def Pphi(r,a,M,theta,alfa,beta):
    return (np.sqrt(gpp(r,a,M,theta))*np.sin(beta)*np.cos(alfa))

@jit(nopython = True)
def Pt(r,a,M,theta,alfa,beta):
    return (-(1+gamma(r,a,M,theta)*np.sqrt(gpp(r,a,M,theta))*                  np.sin(beta)*np.cos(alfa))/(zeta(r,a,M,theta)))

#def dphi(r,a,M,theta,alfa,beta):
#    return (gpp(r,a,M,theta)*Pphi(r,a,M,theta,alfa,beta)+gtp(r,a,M,theta)*Pt(r,a,M,theta,alfa,beta))

#def dt(r,a,M,theta,alfa,beta):
#    return (gtt(r,a,M,theta)*Pt(r,a,M,theta,alfa,beta) + gtp(r,a,M,theta)*Pphi(r,a,M,theta,alfa,beta))

@jit(nopython = True)
def dphi(r,a,M,theta,alfa,beta,p_t,p_phi):
    return (gpp_inv(r,a,M,theta)*p_phi+gtp_inv(r,a,M,theta)*p_t)

@jit(nopython = True)
def dt(r,a,M,theta,alfa,beta,p_t,p_phi):
    return (gtt_inv(r,a,M,theta)*p_t + gtp_inv(r,a,M,theta)*p_phi)

###   COMPUTATION OF CHRISTOFFEL SYMBOLS   ###

@jit(nopython = True)
def C001(r,a,M,theta):
    return (-2*M*((a*M)**2+r**2)*((a*M)**2-2*r**2+(a*M)**2*np.cos(2*theta))/            (((a*M)**2+r*(r-2*M))*((a*M)**2+2*r**2+(a*M)**2*np.cos(2*theta))**2))

@jit(nopython = True)
def C002(r,a,M,theta):
    return (-(4*(a*M)**2*M*r*np.sin(2*theta))/((a*M)**2+2*r**2+(a*M)**2*np.cos(2*theta))**2)

@jit(nopython = True)
def C013(r,a,M,theta):
    return((2*(a*M)*M*((a*M)**4-3*(a*M)**2*r**2-6*r**4+(a*M)**2*((a*M)**2-r**2)*np.cos(2*theta))*np.sin(theta)**2)/           (((a*M)**2+r*(r-2*M))*((a*M)**2+2*r**2+(a*M)**2*np.cos(2*theta))**2))
    

@jit(nopython = True)
def C023(r,a,M,theta):
    return ((8*(a*M)**3*M*r*np.cos(theta)*np.sin(theta)**3)/((a*M)**2+2*r**2+(a*M)**2*np.cos(2*theta))**2)

@jit(nopython = True)
def C100(r,a,M,theta):
    return (M*(-(a*M)**2+(2*M-r)*r)*((a*M)**2*np.cos(theta)**2-r**2)/(r**2+(a*M)**2*np.cos(theta)**2)**3)

@jit(nopython = True)
def C103(r,a,M,theta):
    return (((a*M)*M*((a*M)**2+r*(r-2*M))*(-r**2+(a*M)**2*np.cos(theta)**2)*np.sin(theta)**2/(r**2+(a*M)**2*np.cos(theta)**2)**3))

@jit(nopython = True)
def C111(r,a,M,theta):
    return ((r*((a*M)**2-M*r) + (a*M)**2 *(M-r)*np.cos(theta)**2)/(((a*M)**2+r*(r-2*M))*(r**2+(a*M)**2*np.cos(theta)**2)))

@jit(nopython = True)
def C112(r,a,M,theta):
    return (-((a*M)**2*np.cos(theta)*np.sin(theta)/(r**2+(a*M)**2*np.cos(theta)**2)))

@jit(nopython = True)
def C122(r,a,M,theta):
    return (-r*((a*M)**2+r*(r-2*M))/(r**2+(a*M)**2*np.cos(theta)**2))

@jit(nopython = True)
def C133(r,a,M,theta):
    return (-(((a*M)**2+r*(r-2*M))*np.sin(theta)**2*(r**5+(a*M)**4*r*np.cos(theta)**4-            (a*M)**2*M*r**2*np.sin(theta)**2+np.cos(theta)**2*(2*(a*M)**2*r**3+(a*M)**4*M*np.sin(theta)**2)))/            (r**2+(a*M)**2*np.cos(theta)**2)**3)

@jit(nopython = True)
def C200(r,a,M,theta):
    return (-2*(a*M)**2*M*r*np.cos(theta)*np.sin(theta)/(r**2+(a*M)**2*np.cos(theta)**2)**3)

@jit(nopython = True)
def C203(r,a,M,theta):
    return ((a*M)*M*r*((a*M)**2+r**2)*np.sin(2*theta)/(r**2+(a*M)**2*np.cos(theta)**2)**3)

@jit(nopython = True)
def C211(r,a,M,theta):
    return ((a*M)**2*np.cos(theta)*np.sin(theta)/(((a*M)**2+r*(-2*M+r))*(r**2+(a*M)**2*np.cos(theta)**2)))

@jit(nopython = True)
def C212(r,a,M,theta):
    return (r/(r**2+(a*M)**2*np.cos(theta)**2))

@jit(nopython = True)
def C222(r,a,M,theta):
    return (-((a*M)**2*np.cos(theta)*np.sin(theta)/(r**2+(a*M)**2*np.cos(theta)**2)))

@jit(nopython = True)
def C233(r,a,M,theta):
    return (-(1/(r**2+(a*M)**2*np.cos(theta)**2)**3)*np.cos(theta)*np.sin(theta)*            (2*(a*M)**2*r**2*((a*M)**2+r**2)*np.cos(theta)**2+(a*M)**4*((a*M)**2+r**2)*np.cos(theta)**4+            r*((a*M)**2*r**3+r**5+ 4*(a*M)**2*M*r**2*np.sin(theta)**2+               2*(a*M)**4*M*np.sin(theta)**4+(a*M)**4*M*np.sin(2*theta)**2)))

@jit(nopython = True)
def C301(r,a,M,theta):
    return (((a*M)*M*(r**2-(a*M)**2*np.cos(theta)**2))/(((a*M)**2+r*(r-2*M))*(r**2+(a*M)**2*np.cos(theta)**2)**2))

@jit(nopython = True)
def C302(r,a,M,theta):
    return (-(8*(a*M)*M*r/np.tan(theta))/((a*M)**2+ 2*r**2+(a*M)**2*np.cos(2*theta))**2)

@jit(nopython = True)
def C313(r,a,M,theta):
    return (((a*M)**4*M+ 3*(a*M)**4*r-12*(a*M)**2*M*r**2+8*(a*M)**2*r**3-             16*M*r**4+ 8*r**5+ 4*(a*M)**2*r*((a*M)**2+ r*(2*r-M))*np.cos(2*theta)             -(a*M)**4*(M-r)*np.cos(4*theta))/(2*((a*M)**2 + r*(r-2*M))*((a*M)**2+ 2*r**2+(a*M)**2*np.cos(2*theta))**2))

@jit(nopython = True)
def C323(r,a,M,theta):
    return ((3*(a*M)**4+8*(a*M)**2*M*r+ 8*(a*M)**2*r**2+8*r**4+ 4*(a*M)**2*((a*M)**2+2*r*(r-M))*np.cos(2*theta)              +(a*M)**4*np.cos(4*theta))/(np.tan(theta)*2*((a*M)**2+2*r**2+(a*M)**2*np.cos(2*theta))**2))
            


# In[3]:


@jit(nopython = True)
def run_kut4_mod(F,x,y,h,*args):
    
    hmax = -0.02
    hmin = -10**-7
    #tol = 0.001
    tol = 10**-4
    
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
    
    while error > tol:
        h = delta * h
        
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


    else:
        if np.abs(h) < np.abs(hmin):
            h = hmin
        elif np.abs(h) > np.abs(hmax):
            h = hmax
        else:
            h = delta * h
        
        x = x + h
        y = y4

    #if (delta <= 0.1) :
    #    h = h * 0.1

    #elif (delta >= 4.0 ) :
    #    h = h * 4.0

    #else :
    #    h = delta * h
                   
    return (h,x,y)  


# In[4]:


@jit(nopython = True)
def geo(t,z,a,M,alfa,beta,r0,theta0):

        #R = z[0]
        #dR = z[1]
        #The = z[2]
        #dThe = z[3]

        #dRdt = dR

        #dThedt = dThe
        pphi = Pphi(r0,a,M,theta0,alfa,beta)
        pt = Pt(r0,a,M,theta0,alfa,beta)


        d2Rdt = -(C100(z[0],a,M,z[2])*dt(z[0],a,M,z[2],alfa,beta,pt,pphi)**2 + 2*C103(z[0],a,M,z[2])*dt(z[0],a,M,z[2],alfa,beta,pt,pphi)*dphi(z[0],a,M,z[2],alfa,beta,pt,pphi) +                  C111(z[0],a,M,z[2])*(z[1])**2 + 2*C112(z[0],a,M,z[2])*z[1]*z[3] + C122(z[0],a,M,z[2])*(z[3])**2 + C133(z[0],a,M,z[2])*                 dphi(z[0],a,M,z[2],alfa,beta,pt,pphi)**2)

        d2Thedt = -(C200(z[0],a,M,z[2])*(dt(z[0],a,M,z[2],alfa,beta,pt,pphi))**2 + C203(z[0],a,M,z[2])*dt(z[0],a,M,z[2],alfa,beta,pt,pphi)*dphi(z[0],a,M,z[2],alfa,beta,pt,pphi) +                    C211(z[0],a,M,z[2])*(z[1])**2 + 2*C212(z[0],a,M,z[2])*z[1]*z[3] + C222(z[0],a,M,z[2])*(z[3])**2 + C233(z[0],a,M,z[2])*                   dphi(z[0],a,M,z[2],alfa,beta,pt,pphi)**2)
        
        dPhidt = gtp_inv(z[0],a,M,z[2])*pt + gpp_inv(z[0],a,M,z[2])*pphi
         


        return np.array([z[1], d2Rdt, z[3], d2Thedt,dPhidt])


# In[5]:


@jit(nopython = True)
def func(y,x,h,alfa,beta,a,M,r0,theta0):
    
    it=0
    X=[]
    Y=[]
    X.append(x)
    Y.append(y)
    while y[0] > 1.01*(1+np.sqrt(1-a**2)):
        #y = run_kut4_mod(geo, x, y, h,a,M,alfa,beta,r0,theta0)[2]
        #h = run_kut4_mod(geo, x, y, h,a,M,alfa,beta,r0,theta0)[0]
        #x = run_kut4_mod(geo, x, y, h,a,M,alfa,beta,r0,theta0)[1]
        (h,x,y) = run_kut4_mod(geo, x, y, h,a,M,alfa,beta,r0,theta0)
        it+=1
        X.append(x)
        Y.append(y)
        
        if y[0] > 30:
            yf = [Y[-1][0],Y[-1][2],Y[-1][4]]
            #yf = Y[-1][0]
            break
       
        yf = [Y[-1][0],Y[-1][2],Y[-1][4]]
    #Y0 = [item[0] for item in Y]
    #Y1 = [item[1] for item in Y]
    #Y2 = [item[2] for item in Y]
    #Y3 = [item[3] for item in Y]
    #Y4 = [item[4] for item in Y]
    #return(X,Y0,Y1,Y2,Y3,Y4)    
  
    return (yf)


# In[ ]:


#function to plot the BH shadow
start = time.time()
a = 0.0
M = 1.0
r0 = 15.0
theta0 = np.pi/2
#alfa = np.linspace(np.arctan(10/15),-np.arctan(10/15),88)
#beta = np.linspace(np.arctan(10/15),-np.arctan(10/15),88)

alfaa = np.linspace(-np.arctan(10/15),np.arctan(10/15),752)
betaa = np.linspace(np.arctan(10/15),-np.arctan(10/15),752)



alfa = np.linspace(alfaa[0],alfaa[int(len(alfaa)/2)-1],int(len(alfaa)/2))
beta = np.linspace(betaa[0],betaa[int(len(betaa)/2)-1],int(len(betaa)/2))


Mat = np.zeros((len(alfa),len(beta)))
Mthet = np.zeros((len(alfa),len(beta)))
Mphi = np.zeros((len(alfa),len(beta)))
for i in range(len(alfa)):
    for j in range(len(beta)):
        #start1 = time.time()
        y = np.array([r0,dr(r0,a,M,theta0,alfa[i],beta[j]), theta0,dthe(r0,a,M,theta0,alfa[i]),0])
        #Mat[i,j] = func(y,300.0,-0.02,alfa[i],beta[j],a,M,r0,theta0)[0]
        #Mthet[i,j] = func(y,300.0,-0.02,alfa[i],beta[j],a,M,r0,theta0)[1]
        #Mphi[i,j] = func(y,300.0,-0.02,alfa[i],beta[j],a,M,r0,theta0)[2]
        (Mat[i,j],Mthet[i,j],Mphi[i,j]) = func(y,300.0,-0.02,alfa[i],beta[j],a,M,r0,theta0) 
        #end1 = time.time()
        #print(end1-start1,alfa[i],beta[j])
end = time.time()
print(end-start)

np.savetxt('Mat',Mat)
np.savetxt('Mthet',Mthet)
np.savetxt('Mphi',Mphi)
#np.savetxt('Matrix_mat',Mat)
#np.savetxt('Matrix_thet',Mthet)
#np.savetxt('Matrix_:phi',Mphi)
#Mat1=np.loadtxt("Matrix_mat")
#Mthet1=np.loadtxt("Matrix_thet")
#Mphi1=np.loadtxt("Matrix_phi")


# In[7]:

"""
#algorithm to plot the BH shadow

#np.savetxt('Matrix_mat',Mat)
#np.savetxt('Matrix_theta',Mthet)
#np.savetxt('Matrix_phi',Mphi)
#Mat = np.loadtxt("Matrix_mat")
#Mthet = np.loadtxt("Matrix_thet")
#Mphi = np.loadtxt("Matrix_phi")

a = 0.0

Mphi2 = np.zeros((len(Mphi),len(Mphi[0])))
Mthet2 = np.zeros((len(Mthet),len(Mthet[0])))

for i in range(len(Mthet)):
    for j in range(len(Mthet[0])):
        Mthet2[i,j] = Mthet[i,j]#-int(Mthet[i,j]/(2*np.pi))
        
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
            

dang = 0.01
kmax_the = np.pi/ (10*np.pi/180)
kmax_phi = 2*np.pi/ (10*np.pi/180)
            
M2 = np.zeros((len(Mat),len(Mat[0])))
for i in range(len(Mat)):
    for j in range(len(Mat[0])):
        if Mat[i,j] < 1.01*(1+np.sqrt(1-a**2)):
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
        
        #if (( ((Mthet2[i,j] < np.pi/2 + 10*np.pi/180 and Mthet2[i,j]> np.pi/2-10*np.pi/180) and \
        #     (Mphi2[i,j]< np.pi + 10*np.pi/180 and Mphi2[i,j]> np.pi -10*np.pi/180)) and \
        #   ((Mphi2[i,j]-np.pi)**2 + (Mthet2[i,j]-np.pi/2)**2 <= np.pi * (np.pi*10/180)**2) ) and \
        #   Mat[i,j] >=1.01*(1+np.sqrt(1-a**2))):
        #        M2[i,j] = 5  
        
        kmax = kmax_the
        for k in range(0,int(kmax)+1):
            if (Mthet2[i,j] <= k*10*np.pi/180 + dang and Mthet2[i,j] >= k*10*np.pi/180-dang):
                M2[i,j] = 0

        kmax = kmax_phi
        for k in range(0,int(kmax)+1):
            if (Mphi2[i,j] <= k*10*np.pi/180 + dang and Mphi2[i,j] >= k*10*np.pi/180-dang):
                M2[i,j] = 0
                     
             
            

theta= theta0
a=0.0
#r = np.linspace(r1(a,M),r2(a,M),5000)
plt.figure(figsize = (25,25))

#c_map = [[0,0,0],[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0]]#,[1,1,1]]#,[0,1,1]]
c_map = [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0]]
cm = matplotlib.colors.ListedColormap(c_map)

#plt.imshow(M2, cmap = cm,extent=[-np.sqrt(gpp(r0,a,M,theta))*beta[0],-np.sqrt(gpp(r0,a,M,theta))*beta[-1],np.sqrt(gpp(r0,a,M,theta))*alfa[0],np.sqrt(gpp(r0,a,M,theta))*alfa[-1]])
#plt.imshow(M22, cmap = "binary",extent=[beta[0],beta[-1],alfa[0],alfa[-1]])

plt.imshow(M2, cmap = cm)#,extent=[-beta[0],-beta[-1],alfa[0],alfa[-1]])

#plt.plot(x(r,a,M,theta,r0),yplus(r,a,M,theta,r0),'o',color="red", label="analytical")
#plt.plot(x(r,a,M,theta,r0),yminus(r,a,M,theta,r0),'o',color="red",label = "analytical")

plt.xlabel("x(M)",size=40)
plt.ylabel("y(M)",size=40)

plt.tick_params(axis='both', which='major', labelsize=30)
#plt.legend(loc=10, bbox_to_anchor=(0.85, 0.9), ncol=1,fontsize=17)
#plt.xlim(-np.arctan(10/15),np.arctan(10/15))
#plt.ylim(-np.arctan(10/15),np.arctan(10/15))
plt.show()
#plt.savefig("Kerr_lensing")


# In[11]:
"""

"""
def r1(a,M):
    return (2*(1+np.cos((2/3)*np.arccos(-(np.abs(a*M)/M)))))

def r2(a,M):
    return (2*(1+np.cos((2/3)*np.arccos((np.abs(a*M)/M)))))


def delta(r,a,M):
    return (r**2 + (a*M)**2 - 2*M*r)

def lamb(r,a,M):
    return (-((r/M)**3-3*(r/M)**2+(a*M)**2*r/M + (a*M)**2)/((a*M*(r/M-1))))

def eta(r,a,M):
    return (-(r/M)**3*((r/M)**3-6*(r/M)**2+9*r/M-4*(a*M)**2)/((a*M)**2*(r/M-1)**2))

def yplus(r,a,M,theta,r0):
    return (np.sqrt(gpp(r0,a,M,theta))*np.arcsin((1/(zeta(r0,a,M,theta)-lamb(r,a,M)*gamma(r0,a,M,theta)))*np.sqrt(eta(r,a,M))/np.sqrt(r0**2 +(a*M)**2*np.cos(theta)**2)))

def yminus(r,a,M,theta,r0):
    return (np.sqrt(gpp(r0,a,M,theta))* np.arcsin((-1/(zeta(r0,a,M,theta)-lamb(r,a,M)*gamma(r0,a,M,theta)))*np.sqrt(eta(r,a,M))/np.sqrt(r0**2 +(a*M)**2*np.cos(theta)**2)))
#np.sqrt(gpp(r0,a,M,theta))
def x(r,a,M,theta,r0):
    return (-np.sqrt(gpp(r0,a,M,theta))* np.arctan(lamb(r,a,M)*np.sqrt((r0**2+(a*M)**2*np.cos(theta)**2)*delta(r0,a,M))/(np.sqrt(gpp(r0,a,M,theta))*np.sqrt(r0**4 +((a*M)**2-eta(r,a,M)-lamb(r,a,M)**2)*r0**2 +2*M*r0*(eta(r,a,M)+((a*M)-lamb(r,a,M))**2)-eta(r,a,M)*(a*M)**2))))


# In[14]:


M2 = np.zeros((len(Mat),len(Mat[0])))
count = 0
for i in range(len(Mat)):
    for j in range(len(Mat[0])):
        #if (Mat[i,j] < 0.08 and np.abs(Mz[i,j]) < 1.01):
        if Mat[i,j] < 1.01*(1+np.sqrt(1-0.0**2)):
            M2[i,j] = 1
            count+=1
        else:
            M2[i,j] = 0
print(count)            
plt.figure(figsize = (15,15))


#c_map = [[0,0,0],[1, 1, 1], [1, 0, 0]]#,[1,1,1]]#,[0,1,1]]
#cm = matplotlib.colors.ListedColormap(c_map)

plt.imshow(M2, cmap = "binary",extent=[-beta[0],-beta[-1],alfa[0],alfa[-1]])

plt.xlabel("x(M)",size=30)
plt.ylabel("y(M)",size=30)

plt.tick_params(axis='both', which='major', labelsize=14)
#plt.legend(loc=10, bbox_to_anchor=(0.85, 0.9), ncol=1,fontsize=17)
#plt.xlim(-np.arctan(10/15),np.arctan(10/15))
#plt.ylim(-np.arctan(10/15),np.arctan(10/15))
plt.show()
#plt.savefig("Schwarzschild_Weyl_coords_304x304_adaptive")     


# In[ ]:


"""

