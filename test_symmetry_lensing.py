#!/usr/bin/env python
# coding: utf-8
"""Symmetry-exploiting gravitational-lensing ray tracer for the Schwarzschild
BH + Morgan-Morgan disk system, in Weyl coordinates.

Unlike the plain shadow tracers in this repo (test_Z_SHADOW.py,
test2_Z_SHADOW.py, test_parallel_SHADOW.py), this file builds a
gravitational-lensing *image* rather than just a binary capture/escape
silhouette: its geodesic state vector carries an extra phi component, and its
single-ray tracer `func` returns the photon's final (rho, z, phi), not just
(rho, z). That final phi is what lets downstream plotting distinguish which
patch of sky/disk a given pixel's light ray actually originated from.

To cut compute, this file exploits the up-down and left-right reflection
symmetry of the Weyl-coordinate geometry: the driver at the bottom only
ray-traces one quadrant of the (alfa, beta) emission-angle grid (each of
`alfaa`/`betaa` is halved before tracing). A large, inert, triple-quoted
commented-out block further down contains the intended mirroring/
reconstruction step -- it rebuilds the full four-quadrant matrices from the
traced quadrant via index reflections (with a sign flip for z and a
`2*pi - phi` wrap for the angular coordinate), plus legacy pixel
classification and matplotlib plotting code. None of that block executes.

Structural quirk worth flagging explicitly, same as the canonical file: `nu`/
`nuD` are defined *twice*. The first pair (an earlier "inverted Morgan-Morgan
disk" scheme built on helper coordinates `xx`/`yy`) is entirely inert -- it
lives inside a triple-quoted commented-out block, so it is never executed.
Only the second, `xi2`-based pair is live.

This file depends on a pre-tabulated lambda-potential matrix loaded via
`np.loadtxt("Mat_nu_disk0.1")` (produced by generate_matriz.py), and saves
its output matrices (`Mat`, `Mz`, `Mphi`) via np.savetxt for the plotting
scripts (symmetry.py, simetria_shadow*.py, lensing_image.py) to consume.

This is a straight indentation/documentation pass over the original
Jupyter-notebook export (note the `# In[NN]` cell markers, kept for
provenance); no behaviour was changed.
"""

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
# Numerical utilities: integration and derivatives
@jit(nopython=True)
def simps(f,l,z,a,b,N,*args):
    """Composite Simpson's rule for f(x, z, *args) or f(z, x, *args). #Duplicated: identical/near-identical to simps in generate_matriz.py (canonical copy: test_parallel_SHADOW.py).

    Integrates a 1-D slice of `f` over [a, b] using N (even) subintervals.

    Args:
        f: integrand, called as f(x, z, *args) if l == 0, or f(z, x, *args) if l == 1.
        l: 0 to integrate over the function's first argument, 1 for its second.
        z: the argument held fixed while integrating over the other one.
        a, b: integration limits for the swept argument.
        N: number of subintervals (must be even).
        *args: extra positional arguments forwarded to f.

    Returns:
        The Simpson's-rule estimate of the integral.
    """
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

@jit(nopython=True)
def derivative(f,l,x,y,hder,*args):
    """Central finite-difference derivative of f(x, y, *args). #Duplicated: identical/near-identical to derivative in generate_matriz.py (canonical copy: test_parallel_SHADOW.py).

    Args:
        f: function of two coordinates plus *args.
        l: 0 to differentiate with respect to x, 1 to differentiate with respect to y.
        x, y: point at which to evaluate the derivative.
        hder: finite-difference step size (its absolute value is used).
        *args: extra positional arguments forwarded to f.

    Returns:
        The central-difference approximation of df/dx or df/dy. Prints an
        error and implicitly returns None if l is neither 0 nor 1.
    """
    if l == 0:
        return ( 0.5*(f(x+np.abs(hder),y,*args)-f(x-np.abs(hder),y,*args))/np.abs(hder) )

    elif l == 1:
        return ( 0.5*(f(x,y+np.abs(hder),*args)-f(x,y-np.abs(hder),*args))/np.abs(hder) )

    elif (l != 0 and l != 1):
        print("Error. Check 'l' number")

##################################


# Potentials: BH + Morgan-Morgan disk in Weyl coordinates
@jit(nopython=True)
def d1(rho,z,M):
    """Distance from (rho, z) to the upper Weyl-coordinate rod endpoint (0, +M) of the Schwarzschild BH. #Duplicated: identical/near-identical to d1 in generate_matriz.py (canonical copy: test_parallel_SHADOW.py)."""
    return np.sqrt(rho**2+(z-M)**2)

@jit(nopython=True)
def d2(rho,z,M):
    """Distance from (rho, z) to the lower Weyl-coordinate rod endpoint (0, -M) of the Schwarzschild BH. #Duplicated: identical/near-identical to d2 in generate_matriz.py (canonical copy: test_parallel_SHADOW.py)."""
    return np.sqrt(rho**2+(z+M)**2)


#########             Inverted MORGAN MORGAN DISCS         ###############

# Legacy (dead): earlier "inverted" Morgan-Morgan disk coordinate helpers
# (xx, yy), apparently unused -- entirely superseded by the xi2-based nuD
# below. Kept inert inside a triple-quoted commented-out block.
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


# Legacy (dead/shadowed): earlier nu/nuD pair built on the inverted-disk
# xx/yy coordinates above. Entirely inert (inside this string literal) --
# never executed. #Duplicated (shadowed): superseded by the live xi2-based
# nu/nuD pair defined further below, which wins at import time.
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
# Legacy (dead): an alternate complex-analytic nuD formulation (via
# cmath.sqrt/log), also never executed.
"""
@jit(nopython = True)
def nuD(rho,z,MD,b):
	s = cmath.sqrt(rho**2 + ( (rho**2+z**2) / b*1j-z)**2 )
	mu = -z + (rho**2+z**2) / b *1j + s
	return ( -3*MD*b**2/ (rho**2+z**2)**(5/2) * ( (z**2 - 0.5*rho**2 + (rho**2+z**2)**2/b**2)*cmath.log(mu) + 0.5 * (3*z + (rho**2+z**2)/b*1j) * s ).imag )
"""

@jit(nopython=True)
def xi2(R,z,a):
    """Oblate-spheroidal-like coordinate used in the (live) Morgan-Morgan disk potential nuD, for disk parameter a. #Duplicated: identical/near-identical to xi2 in generate_matriz.py (canonical copy: test_parallel_SHADOW.py)."""
    return ( np.sqrt((a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4))/(R**2 + z**2))/np.sqrt(2) )


@jit(nopython=True)
def nuD(R,z,M,a):
    """Morgan-Morgan finite thin-disk contribution to the nu metric potential (live version). #Duplicated: identical/near-identical to nuD in generate_matriz.py (canonical copy: test_parallel_SHADOW.py).

    Args:
        R, z: Weyl cylindrical coordinates.
        M: disk mass parameter.
        a: disk outer-edge (radius) parameter.

    Returns:
        nuD(R, z), handling the on-disk-plane singular case (xi2 == 0) separately.
    """
    #if (np.abs(z) < 10**-6 and R >= a):
    if xi2(R,z,a) == 0.0:
        #print("a")
        return ( (M*np.sqrt(np.abs(a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))*(-3*a**2 + R**2 + z**2 + 3*np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))/(np.sqrt(2)*np.pi*(R**2 + z**2)**2) - (2*M*(-(a**2*(R**2 - 2*z**2)) + 2*(R**2 + z**2)**2)*np.pi/2)/(np.pi*(R**2 + z**2)**2.5) )
    #if (xi2(R,z,a) == 0.0):
    #	return  (- (2*M*(-(a**2*(R**2 - 2*z**2)) + 2*(R**2 + z**2)**2)*np.pi/2)/(np.pi*(R**2 + z**2)**2.5) )
    else:
        #print("b")
        return ( (M*np.sqrt(np.abs(a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))*(-3*a**2 + R**2 + z**2 + 3*np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4)))/(np.sqrt(2)*np.pi*(R**2 + z**2)**2) - (2*M*(-(a**2*(R**2 - 2*z**2)) + 2*(R**2 + z**2)**2)*np.arctan(np.sqrt(2)*np.sqrt((R**2 + z**2)/np.abs((a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4))))))/(np.pi*(R**2 + z**2)**2.5) )


@jit(nopython=True)
def nu(rho,z,M,MD,b,m): #Schwarzschild \nu potential: nuSch
    """Total nu potential (live version): Schwarzschild BH, Morgan-Morgan disk, or their sum. #Duplicated: identical/near-identical to nu in generate_matriz.py (canonical copy: test_parallel_SHADOW.py).

    Args:
        rho, z: Weyl cylindrical coordinates.
        M: black-hole mass parameter.
        MD: disk mass parameter.
        b: disk radius parameter (passed through to nuD as `a`).
        m: component selector -- 0: BH only (nuSch), 1: disk only (nuD), 2: BH + disk.

    Returns:
        The selected nu potential value.
    """
    #BH
    #DISK

    nuSch = math.log((d1(rho,z,M) + d2(rho,z,M) - 2*M)/(d1(rho,z,M) + d2(rho,z,M) +2*M))

    if m == 0:

        return nuSch

    elif m == 1:

        return nuD(rho,z,MD,b)

    elif m == 2:

        return (nuSch + nuD(rho,z,MD,b))


@jit(nopython=True)
def lambSch(rho,z,M,MD,b): #Schwarzshild \lambda potential: lambSch
    """Closed-form Schwarzschild lambda potential (integration boundary value for lamb_Mat). #Duplicated: identical/near-identical to lambSch in generate_matriz.py (canonical copy: test_parallel_SHADOW.py)."""
    sigma = np.sqrt((rho**2 + z**2 + M**2)**2 - 4*z**2*M**2)
    return np.log( ((d1(rho,z,M) + d2(rho,z,M))**2-4*M**2)/(4*sigma) )


#####################################################################


#####################################################################

########      TO CALCULATE THE INITIAL OBSERVER'S RHO       #########

# Observer-frame helpers: non-jitted copies used only to solve for the
# initial rho via fsolve (below the module-scope driver).
def d1_i(rho,z,M):
    """Observer-frame (non-jitted) copy of d1, used only to locate the initial observer position via fsolve. #Duplicated: identical/near-identical to d1_i in generate_matriz.py (canonical copy: test_parallel_SHADOW.py)."""
    return np.sqrt(rho**2+(z-M)**2)


def d2_i(rho,z,M):
    """Observer-frame (non-jitted) copy of d2. #Duplicated: identical/near-identical to d2_i in generate_matriz.py (canonical copy: test_parallel_SHADOW.py)."""
    return np.sqrt(rho**2+(z+M)**2)




def xi2_i(R,z,a):
    """Observer-frame (non-jitted) copy of xi2. #Duplicated: identical/near-identical to xi2_i in generate_matriz.py (canonical copy: test_parallel_SHADOW.py)."""
    return ( np.sqrt((a**2 - R**2 - z**2 + np.sqrt((a**2 - R**2)**2 + 2*(a**2 + R**2)*z**2 + z**4))/(R**2 + z**2))/np.sqrt(2) )



def nuD_i(R,z,M,a):
    """Observer-frame (non-jitted) copy of nuD. #Duplicated: identical/near-identical to nuD_i in generate_matriz.py (canonical copy: test_parallel_SHADOW.py)."""
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
    """Observer-frame (non-jitted) copy of nu, used only to locate the initial observer position via fsolve. #Duplicated: identical/near-identical to nu_i in generate_matriz.py (canonical copy: test_parallel_SHADOW.py).

    #NOTE: the m == 1 branch calls the jitted `nuD`, not this file's own
    `nuD_i` -- kept as in the original (same pre-existing quirk as the
    canonical file).
    """
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
    """Observer-frame (non-jitted) copy of gpp, used to solve for the initial observer's rho via fsolve. #Duplicated: identical/near-identical to gpp_i in generate_matriz.py (canonical copy: test_parallel_SHADOW.py)."""
    return rho**2 * math.exp(- nu_i(rho,z,M,MD,b,2))


#####################################################################
#####################################################################
#####################################################################


# Lambda interpolation: finite-difference derivatives feeding the
# vacuum Weyl field equations, plus bilinear interpolation of the
# pre-tabulated lambda matrix.
@jit(nopython=True)
def derNU(rho,z,M,MD,b,l,m,hder):
    """Finite-difference derivative of `nu` with respect to rho (l=0) or z (l=1), component m. #Duplicated: identical/near-identical to derNU in generate_matriz.py (canonical copy: test_parallel_SHADOW.py)."""
    return derivative(nu,l,rho,z,hder,M,MD,b,m)


@jit(nopython=True)
def dlamb(z,rho,M,MD,b,l,m,hder):
    """Integrand for lambda's z-derivative (l=1) or rho-derivative (l=0), from the vacuum Weyl field equations.

    #Duplicated: identical/near-identical to dlamb in generate_matriz.py
    (canonical copy: test_parallel_SHADOW.py). #NOTE: identical body to
    `dlamb2` below -- the only difference is argument order, (z, rho, ...)
    here so `sci.quad` can integrate over z while treating rho as a fixed
    parameter.
    """
    if l == 0:
        return 0.5*rho * ( derNU(rho,z,M,MD,b,0,m,hder)**2 - derNU(rho,z,M,MD,b,1,m,hder)**2)

    elif l == 1:
        return rho * derNU(rho,z,M,MD,b,0,m,hder) * derNU(rho,z,M,MD,b,1,m,hder)

@jit(nopython=True)
def dlamb2(rho,z,M,MD,b,l,m,hder):
    """Same integrand as `dlamb`, but with (rho, z) argument order so `sci.quad` can integrate over rho.

    #Duplicated: identical/near-identical to dlamb2 in generate_matriz.py
    (canonical copy: test_parallel_SHADOW.py); identical body to `dlamb`
    (only the parameter order differs).
    """
    if l == 0:
        return 0.5*rho * ( derNU(rho,z,M,MD,b,0,m,hder)**2 - derNU(rho,z,M,MD,b,1,m,hder)**2)

    elif l == 1:
        return rho * derNU(rho,z,M,MD,b,0,m,hder) * derNU(rho,z,M,MD,b,1,m,hder)


# Legacy (dead): earlier ground-truth lambda(rho,z) quadrature (lamb_Mat),
# entirely superseded here by the bilinear-interpolation `lamb` below, which
# reads from the pre-tabulated matrix produced by generate_matriz.py instead
# of re-integrating on every call. #Duplicated (shadowed): the live version
# of this function is what generate_matriz.py actually executes to build
# that matrix.
"""@jit
def lamb_Mat(rho,z,M,MD,b,m,hder):
    limit = 0.0
    if z >= limit:
        return sci.quad(dlamb,40.0,z,(rho,M,MD,b,1,m,hder))[0] + sci.quad(dlamb2,40.0,rho,(40.0,M,MD,b,0,m,hder))[0] + lambSch(40.0,40.0,M+MD,MD,b)

    elif z < -limit:
        return sci.quad(dlamb,-40,z,(rho,M,MD,b,1,m,hder))[0] + sci.quad(dlamb2,40.0,rho,(-40.0,M,MD,b,0,m,hder))[0] + lambSch(40.0,-40.0,M+MD,MD,b)

"""
@jit(nopython=True)
def lamb(rho,z,M,MD,b,m):
    """Bilinear interpolation of lambda(rho, z) from the preloaded matrix `Mat_nu`. #Duplicated: identical/near-identical to lamb in generate_matriz.py (canonical copy: test_parallel_SHADOW.py).

    `Mat_nu` (loaded at module scope from "Mat_nu_disk0.1", produced by
    generate_matriz.py's `lamb_Mat` quadrature) is indexed on a
    zref=Rref=40 grid; (i0, j0) are the integer grid indices below the
    target point and (I, J) are the fractional interpolation weights within
    that grid cell.

    Args:
        rho, z: target Weyl coordinates.
        M, MD, b, m: unused here except implicitly via the preloaded matrix
            (kept in the signature for interface parity with `nu`/`lamb_Mat`).

    Returns:
        Bilinearly-interpolated lambda(rho, z).
    """

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
# Legacy (dead): earlier inline version of generate_matriz.py's grid-
# tabulation driver, superseded by the standalone generate_matriz.py script;
# never executed here.
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



# Load the pre-tabulated lambda-potential matrix produced by
# generate_matriz.py; `lamb()` above bilinearly interpolates into it.
Mat_nu = np.loadtxt("Mat_nu_disk0.1")


# In[32]:


# Metric components (BH + disk, m=2)
@jit(nopython=True)
def gtt(rho,z,M,MD,b):
    """g_tt metric component at (rho, z). #Duplicated: identical/near-identical to gtt in generate_matriz.py (canonical copy: test_parallel_SHADOW.py)."""
    return -math.exp(nu(rho,z,M,MD,b,2))



@jit(nopython=True)
def grr(rho,z,M,MD,b):
    """g_rho,rho metric component at (rho, z). #Duplicated: identical/near-identical to grr in generate_matriz.py (canonical copy: test_parallel_SHADOW.py)."""
    return math.exp( lamb(rho,z,M,MD,b,2) - nu(rho,z,M,MD,b,2) )

@jit(nopython=True)
def gzz(rho,z,M,MD,b):
    """g_zz metric component at (rho, z). #Duplicated: identical/near-identical to gzz in generate_matriz.py (canonical copy: test_parallel_SHADOW.py)."""
    return math.exp( lamb(rho,z,M,MD,b,2) - nu(rho,z,M,MD,b,2) )

@jit(nopython=True)
def gpp(rho,z,M,MD,b):
    """g_phi,phi metric component at (rho, z). #Duplicated: identical/near-identical to gpp in generate_matriz.py (canonical copy: test_parallel_SHADOW.py)."""
    return rho**2 * math.exp(- nu(rho,z,M,MD,b,2))

##########################

@jit(nopython=True)
def zeta(rho,z,M,MD,b):
    """Redshift-like normalization factor, sqrt(-1/g_tt), at (rho, z). #Duplicated: identical/near-identical to zeta in generate_matriz.py (canonical copy: test_parallel_SHADOW.py)."""
    return np.sqrt(-1/gtt(rho,z,M,MD,b))

###############

# Momenta & initial velocities, from photon emission angles (alfa, beta)
@jit(nopython=True)
def dthe(rho,z,M,MD,b,alfa):
    """Initial d(z)/d(affine parameter)-like "theta" velocity component from emission angle alfa. #Duplicated: identical/near-identical to dthe in generate_matriz.py (canonical copy: test_parallel_SHADOW.py)."""
    return 1/np.sqrt(gzz(rho,z,M,MD,b))*np.sin(alfa)

@jit(nopython=True)
def dr(rho,z,M,MD,b,alfa,beta):
    """Initial drho/d(affine parameter) velocity component from emission angles alfa, beta. #Duplicated: identical/near-identical to dr in generate_matriz.py (canonical copy: test_parallel_SHADOW.py)."""
    return 1/np.sqrt(grr(rho,z,M,MD,b))*np.cos(alfa)*np.cos(beta)

@jit(nopython=True)
def Pphi(rho,z,M,MD,b,alfa,beta):
    """Conserved photon angular momentum from local emission angles (alfa, beta). #Duplicated: identical/near-identical to Pphi in generate_matriz.py (canonical copy: test_parallel_SHADOW.py)."""
    return np.sqrt(gpp(rho,z,M,MD,b))*np.sin(beta)*np.cos(alfa)

@jit(nopython=True)
def Pt(rho,z,M,MD,b,alfa,beta):
    """Conserved photon energy at the initial point. #Duplicated: identical/near-identical to Pt in generate_matriz.py (canonical copy: test_parallel_SHADOW.py)."""
    return -1/zeta(rho,z,M,MD,b)

@jit(nopython=True)
def dphi(rho,z,M,MD,b,alfa,beta,p_phi):
    """d(phi)/d(affine parameter) from the conserved angular momentum p_phi. #Duplicated: identical/near-identical to dphi in generate_matriz.py (canonical copy: test_parallel_SHADOW.py)."""
    return 1/gpp(rho,z,M,MD,b)*p_phi

@jit(nopython=True)
def dt(rho,z,M,MD,b,alfa,beta,p_t):
    """dt/d(affine parameter) from the conserved energy p_t. #Duplicated: identical/near-identical to dt in generate_matriz.py (canonical copy: test_parallel_SHADOW.py)."""
    return 1/gtt(rho,z,M,MD,b)*p_t


# In[33]:


# Integrator
@jit(nopython=True)
def run_kut4_mod(F,x,y,h,*args):
    """Adaptive-step embedded Runge-Kutta-Fehlberg 4(5) (RKF45) integrator step. #Duplicated: identical/near-identical to run_kut4_mod in generate_matriz.py (canonical copy: test_parallel_SHADOW.py).

    Advances the ODE system y' = F(x, y, *args) by one step, estimating
    local error from the 4th- and 5th-order solutions (y4, y5) and shrinking
    `h` (bounded by hmin/hmax) until the error tolerance `tol` is met.

    Args:
        F: right-hand-side function, called as F(x, y, *args).
        x: current independent variable.
        y: current state vector.
        h: current (signed) step size.
        *args: extra positional arguments forwarded to F.

    Returns:
        (h, x, y): the (possibly reduced) step size actually used, and the
        new (x, y) after advancing by that step.
    """

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

    # Legacy (dead): earlier step-refinement loop with a hard iteration cap
    # (it > 300) and verbose per-iteration diagnostics, superseded by the
    # else-branch loop actually used below (it > 500 cap, no printing).
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




# Geodesic & ray tracing
@jit(nopython=True)
def geo(t,z,M,alfa,beta,rho0,z0,MD,b,hder):
    """Geodesic equations of motion (right-hand side), via the Weyl nu/lambda potentials and their derivatives. #Duplicated (variant): differs from the canonical version in test_parallel_SHADOW.py by carrying phi in the state vector: `z` here is [rho, drho, z, dz, phi] (5 components) and the return includes the extra `dPhidt` row -- needed so the final phi is available for the lensing map, whereas the canonical shadow-only `geo` tracks only [rho, drho, z, dz] (4 components).

    Args:
        t: current affine parameter (unused directly, kept for the RKF45 interface).
        z: state vector [rho, drho/dtau, z_coord, dz/dtau, phi].
        M, MD, b: BH mass, disk mass, disk radius.
        alfa, beta: photon emission angles at the initial point, used (with
            rho0, z0) to compute the conserved momenta pphi, pt.
        rho0, z0: initial emission point (kept fixed since pphi, pt are
            constants of motion).
        hder: finite-difference step used inside derNU.

    Returns:
        np.array([drho, d2rho, dz, d2z, dphi]) -- the derivative of the
        state vector z, i.e. the 2nd-order radial/z photon equations plus
        the phi "velocity", derived from the Weyl metric potentials.
    """

    pphi = Pphi(rho0,z0,M,MD,b,alfa,beta)
    pt = Pt(rho0,z0,M,MD,b,alfa,beta)

    d2Rdt = -(0.5*math.exp(2*nu(z[0],z[2],M,MD,b,2)-lamb(z[0],z[2],M,MD,b,2))*derNU(z[0],z[2],M,MD,b,0,2,hder)* dt(z[0],z[2],M,MD,b,alfa,beta,pt)**2              +0.5*(dlamb2(z[0],z[2],M,MD,b,0,2,hder)-derNU(z[0],z[2],M,MD,b,0,2,hder))*z[1]**2 +               (dlamb2(z[0],z[2],M,MD,b,1,2,hder)-derNU(z[0],z[2],M,MD,b,1,2,hder))*z[1]*z[3]              -0.5*(dlamb2(z[0],z[2],M,MD,b,0,2,hder)-derNU(z[0],z[2],M,MD,b,0,2,hder))*z[3]**2              +0.5*math.exp(-lamb(z[0],z[2],M,MD,b,2))*z[0]*(-2+z[0]*derNU(z[0],z[2],M,MD,b,0,2,hder))*dphi(z[0],z[2],M,MD,b,alfa,beta,pphi)**2)

    d2Thedt = -(0.5*math.exp(2*nu(z[0],z[2],M,MD,b,2)-lamb(z[0],z[2],M,MD,b,2))*derNU(z[0],z[2],M,MD,b,1,2,hder)* dt(z[0],z[2],M,MD,b,alfa,beta,pt)**2                -0.5*(dlamb2(z[0],z[2],M,MD,b,1,2,hder)-derNU(z[0],z[2],M,MD,b,1,2,hder))*z[1]**2                + (dlamb2(z[0],z[2],M,MD,b,0,2,hder)-derNU(z[0],z[2],M,MD,b,0,2,hder))*z[1]*z[3]                + 0.5*(dlamb2(z[0],z[2],M,MD,b,1,2,hder)-derNU(z[0],z[2],M,MD,b,1,2,hder))*z[3]**2                + 0.5*math.exp(-lamb(z[0],z[2],M,MD,b,2))*z[0]**2*derNU(z[0],z[2],M,MD,b,1,2,hder)*dphi(z[0],z[2],M,MD,b,alfa,beta,pphi)**2)

    dPhidt = 1/gpp(z[0], z[2],M,MD,b)*pphi



    return np.array([z[1], d2Rdt, z[3], d2Thedt,dPhidt])





@jit(nopython=True)
def func(y,x,h,alfa,beta,M,rho0,z0,MD,b,hder):
    """Single-ray tracer: integrates one photon's geodesic until it escapes, is captured, or exits otherwise. #Duplicated (variant): differs from the canonical version in test_parallel_SHADOW.py by returning the photon's final phi as well: `yf = [rho, z, phi]` (3 elements) here, vs. `[rho, z]` (2 elements) in the canonical shadow-only `func` -- this is exactly what turns the tracer into a lensing-map builder rather than a binary shadow tracer.

    Repeatedly advances the state with `run_kut4_mod` while the photon's nu
    potential stays above -3.0 and its areal radius stays below 30 (i.e.
    while it is neither captured nor escaped).

    Args:
        y: initial state [rho0, drho, z0, dz, phi].
        x: initial affine parameter.
        h: initial (negative) step size.
        alfa, beta: emission angles.
        M, rho0, z0, MD, b, hder: BH mass, initial emission point, disk
            mass, disk radius, finite-difference step.

    Returns:
        [rho, z, phi] at the escape point (areal radius >= 30), or the
        sentinel [0.001, 0.001, 0.2] if the photon is captured (nu <= -3.0).
    """

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




# Driver: solve for the initial observer's rho, build the emission-angle
# grid but only over a quarter of the full (alfa, beta) range (symmetry
# exploitation: the full lensing image can be reconstructed from one
# quadrant by reflection, done in the dead mirroring block further below),
# ray-trace that quadrant with `func`, and save the resulting Mat/Mz/Mphi
# matrices.
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

# Legacy plotting: mirroring/reconstruction block (dead, never executed).
# This is the symmetry step referenced above -- it rebuilds full four-
# quadrant Mat2/Mz2/Mphi3 matrices from the traced quadrant via reflected
# indexing ([i,j], [i,-j-1], [-i-1,j], [-i-1,-j-1]), with a sign flip for z
# and a 2*pi-phi wrap for the angular coordinate, then classifies pixels
# into four lensing-image quadrants (M2) and plots them.
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

# Legacy plotting: second (R, theta)-based lensing classification + grid-
# line overlay + figure attempt, also dead/never executed.
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

