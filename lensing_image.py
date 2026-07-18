import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import time
from numba import jit
import cmath
import scipy.integrate as sci
from scipy.optimize import fsolve

alfa = np.linspace(-np.arctan(10/15),np.arctan(10/15),1000)
beta = np.linspace(np.arctan(10/15),-np.arctan(10/15),1000)

Mat = np.loadtxt("Mat")
Mz = np.loadtxt("Mz")
Mphi = np.loadtxt("Mphi")

M2 = np.zeros((len(Mat),len(Mat[0])))
Mz2 = np.zeros((len(Mz),len(Mz[0])))
Mphi2 = np.zeros((len(Mphi),len(Mphi[0])))

M = 0.9
MD = 0.1

dang = 0.01
dz = 0.06
kmax_z   = 18.0 
kmax_phi = 2*np.pi/ (10*np.pi/180)
        
for i in range(len(Mphi)):
	for j in range(len(Mphi)):
        	if Mphi[i,j] > 2*np.pi:
            		Mphi2[i,j] = Mphi[i,j]-int(Mphi[i,j]/(2*np.pi))*2*np.pi
            
        	elif (Mphi[i,j] < 0 and Mphi[i,j] > -2*np.pi):
            		Mphi2[i,j] = Mphi[i,j] + 2*np.pi
            
        	elif Mphi[i,j] < -2*np.pi:
            		Mphi2[i,j] = Mphi[i,j] - int(Mphi[i,j]/(2*np.pi))*2*np.pi + 2*np.pi
            
        	else:
            		Mphi2[i,j] = Mphi[i,j]
            

            
M2 = np.zeros((len(Mat),len(Mat[0])))
for i in range(len(Mat)):
	for j in range(len(Mat[0])):
        	if Mat[i,j] < 0.002:
            		M2[i,j] = 0
        	else:
            		if (Mz[i,j] < 0 and (Mphi2[i,j] >= 0 and Mphi2[i,j] <= np.pi)):
                		M2[i,j] = 3 #1
                
            		elif ( Mz[i,j] < 0 and (Mphi2[i,j] > np.pi and Mphi2[i,j] <= 2*np.pi)):   
                		M2[i,j] = 4 #2
                
            		elif (Mz[i,j] >= 0 and (Mphi2[i,j] > 0  and Mphi2[i,j] <= np.pi)):
                		M2[i,j] = 2 #3
                
            		elif ( Mz[i,j] >= 0 and (Mphi2[i,j] > np.pi and Mphi2[i,j] <= 2*np.pi )):
                		M2[i,j] = 1 #4
   

###################         FIELD LINES            ###############################         
"""
for i in range(len(Mat)):
	for j in range(len(Mat[0])):
        	"Einstein ring"
		#if (( ((Mthet2[i,j] < np.pi/2 + 10*np.pi/180 and Mthet2[i,j]> np.pi/2-10*np.pi/180) and \
		#     (Mphi2[i,j]< np.pi + 10*np.pi/180 and Mphi2[i,j]> np.pi -10*np.pi/180)) and \
		#   ((Mphi2[i,j]-np.pi)**2 + (Mthet2[i,j]-np.pi/2)**2 <= np.pi * (np.pi*10/180)**2) ) and \
		#   Mat[i,j] >=0.02):
		#        M2[i,j] = 5  

        	"Black lines of the grid"
        	kmax = kmax_z
        	for k in range(0,int(kmax)+1):
            		if (Mz[i,j] <= 30 - k*60.0/18.0 + dz and Mz[i,j] >= 30 - k*60.0/18.0 - dz):
                		M2[i,j] = 0
        
        	kmax = kmax_phi
        	for k in range(0,int(kmax)+1):
            		if (Mphi2[i,j] <= k*10*np.pi/180 + dang and Mphi2[i,j] >= k*10*np.pi/180-dang):
                		M2[i,j] = 0
                                  
"""
   


plt.figure(figsize = (25,25))

#c_map = [[0,0,0],[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0]]#,[1,1,1]]#,[0,1,1]]
c_map = [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0]]

cm = matplotlib.colors.ListedColormap(c_map)

#plt.imshow(M2, cmap = cm,extent=[-np.sqrt(gpp(r0,a,M,theta))*beta[0],-np.sqrt(gpp(r0,a,M,theta))*beta[-1],np.sqrt(gpp(r0,a,M,theta))*alfa[0],np.sqrt(gpp(r0,a,M,theta))*alfa[-1]])
#plt.imshow(M22, cmap = "binary",extent=[beta[0],beta[-1],alfa[0],alfa[-1]])

plt.imshow(M2, cmap = cm,extent=[-beta[0],-beta[-1],alfa[0],alfa[-1]])


plt.title("$M_{BN}=$" + str(M) + " , $M_{Disk} =$" + str(MD) + " , a = 6",size=40)
plt.xlabel("$\\beta$",size=30)
plt.ylabel("$\\alpha$",size=30)

plt.tick_params(axis='both', which='major', labelsize=30)
#plt.show()
plt.savefig("Schwarzschild_WeylCoords_lensing_1000x1000")


