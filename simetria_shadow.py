import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors

import matplotlib.patches as mpatches


Mat = np.loadtxt("Mat")
Mz = np.loadtxt("Mz")

M = 0.4
MD = 0.6
b = 3.0

alfa = np.linspace(-np.arctan(10/15),np.arctan(10/15),1000)
beta = np.linspace(np.arctan(10/15),-np.arctan(10/15),1000)


Mat2 = np.zeros(( 2*len(Mat),2*len(Mat[0]) ))
Mz2 = np.zeros(( 2*len(Mz),2*len(Mz[0]) ))


for i in range(len(Mat)):
	for j in range(len(Mat[0])):
		Mat2[i,j] = Mat[i,j]
		Mat2[i,-j-1] = Mat[i,j]
		Mat2[-i-1,j] = Mat[i,j]
		Mat2[-i-1,-j-1] = Mat[i,j]

		##Mz
		Mz2[i,j] = Mz[i,j]
		Mz2[i,-j-1] = Mz[i,j]
		Mz2[-i-1,j] = - Mz[i,j]
		Mz2[-i-1,-j-1] = - Mz[i,j]


M2 = np.zeros((len(Mat2),len(Mat2[0])))

for i in range(len(Mat2)):
	for j in range(len(Mat2[0])):

		if (Mat2[i,j] <= 0.002):# and Mz[i,j] < M):
    			M2[i,j] = 1

		elif ( Mat2[i,j] > b and (Mz2[i,j] > 49.0 or Mz2[i,j] < -49.0)):
			M2[i,j] = 2    
	  	
		
		else:
    			M2[i,j] = 0

## FIGURE ##

#plt.figure(figsize = (22,22))
fig,ax = plt.subplots(1,1,figsize=(30,30))


#c_map = [[1,1,1],[0, 0, 0], [0.192, 0.192, 0.192]]
c_map = [[0.505, 0.505, 0.505],[0, 0, 0], [1,1,1]]


cm = matplotlib.colors.ListedColormap(c_map)



plt.imshow(M2, cmap = cm,extent=[-beta[0],-beta[-1],alfa[0],alfa[-1]])
ax.add_patch(mpatches.Circle((0.006,0.001),0.005,color='blue'))
ax.add_patch(mpatches.Circle((0.006,0.100),0.005,color='orange'))
ax.add_patch(mpatches.Circle((0.006,0.207),0.005,color='green'))
ax.add_patch(mpatches.Circle((0.006,0.220),0.005,color='red'))
ax.add_patch(mpatches.Circle((0.006,0.300),0.005,color='purple'))


#plt.xlabel("$\\beta$",size=84)
#plt.ylabel("$\\alpha$",size=84)

plt.ylim(0.0,np.arctan(10/15))
#plt.tick_params(axis='both', which='major', labelsize=60)

plt.title("$m=$" + str(round(MD,1)),size=84)

plt.show()
#plt.savefig("M=0_4")


