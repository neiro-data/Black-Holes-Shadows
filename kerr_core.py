"""Kerr-metric (r, theta) physics core: metric, Christoffel symbols, and null-geodesic tracer.

Distinct spacetime from the Weyl-coordinate Schwarzschild+disk family in `weyl_core.py`.
Uses Boyer-Lindquist-like (r, theta, phi) coordinates with spin parameter `a` (in units of
M, i.e. the physical spin is `a*M`). Originally embedded directly in
`lensing_r_theta.py`; moved here so that script can become a thin driver.
"""

import numpy as np
from numba import jit, prange


# ---------------------------------------------------------------------------
# Metric components (Boyer-Lindquist-like coordinates), |P| = 1 normalization assumed
# ---------------------------------------------------------------------------
@jit(nopython=True)
def gtt(r, a, M, theta):
    return -(1 - 2 * M * r / (r**2 + (a * M) ** 2 * np.cos(theta) ** 2))


@jit(nopython=True)
def gtp(r, a, M, theta):
    return -2 * M * (a * M) * r * (np.sin(theta)) ** 2 / (r**2 + (a * M) ** 2 * np.cos(theta) ** 2)


@jit(nopython=True)
def gpp(r, a, M, theta):
    return (np.sin(theta)) ** 2 * (
        r**2
        + (a * M) ** 2
        + 2
        * M
        * r
        * (a * M) ** 2
        * (np.sin(theta)) ** 2
        / (r**2 + (a * M) ** 2 * np.cos(theta) ** 2)
    )


@jit(nopython=True)
def grr(r, a, M, theta):
    return (r**2 + (a * M) ** 2 * np.cos(theta) ** 2) / (r**2 - 2 * M * r + (a * M) ** 2)


@jit(nopython=True)
def gthe(r, a, M, theta):
    return r**2 + (a * M) ** 2 * np.cos(theta) ** 2


@jit(nopython=True)
def gpp_inv(r, a, M, theta):
    return (
        ((a * M) ** 2 + 2 * r * (r - 2 * M) + (a * M) ** 2 * np.cos(2 * theta))
        * (1 / np.sin(theta) ** 2)
        / (
            ((a * M) ** 2 + r * (r - 2 * M))
            * ((a * M) ** 2 + 2 * r**2 + (a * M) ** 2 * np.cos(2 * theta))
        )
    )


@jit(nopython=True)
def gtt_inv(r, a, M, theta):
    return -(
        (a * M) ** 4
        + 2 * r**4
        + (a * M) ** 2 * r * (2 * M + 3 * r)
        + (a * M) ** 2 * ((a * M) ** 2 + r * (r - 2 * M)) * np.cos(2 * theta)
    ) / (
        ((a * M) ** 2 + r * (r - 2 * M))
        * ((a * M) ** 2 + 2 * r**2 + (a * M) ** 2 * np.cos(2 * theta))
    )


@jit(nopython=True)
def gtp_inv(r, a, M, theta):
    return (
        -4
        * (a * M)
        * M
        * r
        / (
            ((a * M) ** 2 + r * (r - 2 * M))
            * ((a * M) ** 2 + 2 * r**2 + (a * M) ** 2 * np.cos(2 * theta))
        )
    )


# ---------------------------------------------------------------------------
# Auxiliary observer-frame quantities
# ---------------------------------------------------------------------------
@jit(nopython=True)
def gamma(r, a, M, theta):
    return (-gtp(r, a, M, theta) / gpp(r, a, M, theta)) * np.sqrt(
        gpp(r, a, M, theta) / (gtp(r, a, M, theta) ** 2 - gtt(r, a, M, theta) * gpp(r, a, M, theta))
    )


@jit(nopython=True)
def zeta(r, a, M, theta):
    return np.sqrt(
        gpp(r, a, M, theta) / (gtp(r, a, M, theta) ** 2 - gtt(r, a, M, theta) * gpp(r, a, M, theta))
    )


@jit(nopython=True)
def delta(r, a, M):
    return r**2 - 2 * M * r + (a * M) ** 2


# ---------------------------------------------------------------------------
# Coordinate derivatives / initial momenta
# ---------------------------------------------------------------------------
@jit(nopython=True)
def dthe(r, a, M, theta, alfa):
    return np.sqrt(gthe(r, a, M, theta)) * np.sin(alfa) / (r**2 + (a * M) ** 2 * np.cos(theta) ** 2)


@jit(nopython=True)
def dr(r, a, M, theta, alfa, beta):
    return (
        np.sqrt(grr(r, a, M, theta))
        * np.cos(beta)
        * np.cos(alfa)
        * (r**2 - 2 * M * r + (a * M) ** 2)
        / (r**2 + (a * M) ** 2 * np.cos(theta) ** 2)
    )


@jit(nopython=True)
def Pphi(r, a, M, theta, alfa, beta):
    return np.sqrt(gpp(r, a, M, theta)) * np.sin(beta) * np.cos(alfa)


@jit(nopython=True)
def Pt(r, a, M, theta, alfa, beta):
    return -(
        1 + gamma(r, a, M, theta) * np.sqrt(gpp(r, a, M, theta)) * np.sin(beta) * np.cos(alfa)
    ) / (zeta(r, a, M, theta))


@jit(nopython=True)
def dphi(r, a, M, theta, alfa, beta, p_t, p_phi):
    return gpp_inv(r, a, M, theta) * p_phi + gtp_inv(r, a, M, theta) * p_t


@jit(nopython=True)
def dt(r, a, M, theta, alfa, beta, p_t, p_phi):
    return gtt_inv(r, a, M, theta) * p_t + gtp_inv(r, a, M, theta) * p_phi


# ---------------------------------------------------------------------------
# Christoffel symbols (nonzero components for this metric, axisymmetric + stationary)
# ---------------------------------------------------------------------------
@jit(nopython=True)
def C001(r, a, M, theta):
    return (
        -2
        * M
        * ((a * M) ** 2 + r**2)
        * ((a * M) ** 2 - 2 * r**2 + (a * M) ** 2 * np.cos(2 * theta))
        / (
            ((a * M) ** 2 + r * (r - 2 * M))
            * ((a * M) ** 2 + 2 * r**2 + (a * M) ** 2 * np.cos(2 * theta)) ** 2
        )
    )


@jit(nopython=True)
def C002(r, a, M, theta):
    return (
        -(4 * (a * M) ** 2 * M * r * np.sin(2 * theta))
        / ((a * M) ** 2 + 2 * r**2 + (a * M) ** 2 * np.cos(2 * theta)) ** 2
    )


@jit(nopython=True)
def C013(r, a, M, theta):
    return (
        2
        * (a * M)
        * M
        * (
            (a * M) ** 4
            - 3 * (a * M) ** 2 * r**2
            - 6 * r**4
            + (a * M) ** 2 * ((a * M) ** 2 - r**2) * np.cos(2 * theta)
        )
        * np.sin(theta) ** 2
    ) / (
        ((a * M) ** 2 + r * (r - 2 * M))
        * ((a * M) ** 2 + 2 * r**2 + (a * M) ** 2 * np.cos(2 * theta)) ** 2
    )


@jit(nopython=True)
def C023(r, a, M, theta):
    return (8 * (a * M) ** 3 * M * r * np.cos(theta) * np.sin(theta) ** 3) / (
        (a * M) ** 2 + 2 * r**2 + (a * M) ** 2 * np.cos(2 * theta)
    ) ** 2


@jit(nopython=True)
def C100(r, a, M, theta):
    return (
        M
        * (-((a * M) ** 2) + (2 * M - r) * r)
        * ((a * M) ** 2 * np.cos(theta) ** 2 - r**2)
        / (r**2 + (a * M) ** 2 * np.cos(theta) ** 2) ** 3
    )


@jit(nopython=True)
def C103(r, a, M, theta):
    return (
        (a * M)
        * M
        * ((a * M) ** 2 + r * (r - 2 * M))
        * (-(r**2) + (a * M) ** 2 * np.cos(theta) ** 2)
        * np.sin(theta) ** 2
        / (r**2 + (a * M) ** 2 * np.cos(theta) ** 2) ** 3
    )


@jit(nopython=True)
def C111(r, a, M, theta):
    return (r * ((a * M) ** 2 - M * r) + (a * M) ** 2 * (M - r) * np.cos(theta) ** 2) / (
        ((a * M) ** 2 + r * (r - 2 * M)) * (r**2 + (a * M) ** 2 * np.cos(theta) ** 2)
    )


@jit(nopython=True)
def C112(r, a, M, theta):
    return -(
        (a * M) ** 2 * np.cos(theta) * np.sin(theta) / (r**2 + (a * M) ** 2 * np.cos(theta) ** 2)
    )


@jit(nopython=True)
def C122(r, a, M, theta):
    return -r * ((a * M) ** 2 + r * (r - 2 * M)) / (r**2 + (a * M) ** 2 * np.cos(theta) ** 2)


@jit(nopython=True)
def C133(r, a, M, theta):
    return (
        -(
            ((a * M) ** 2 + r * (r - 2 * M))
            * np.sin(theta) ** 2
            * (
                r**5
                + (a * M) ** 4 * r * np.cos(theta) ** 4
                - (a * M) ** 2 * M * r**2 * np.sin(theta) ** 2
                + np.cos(theta) ** 2
                * (2 * (a * M) ** 2 * r**3 + (a * M) ** 4 * M * np.sin(theta) ** 2)
            )
        )
        / (r**2 + (a * M) ** 2 * np.cos(theta) ** 2) ** 3
    )


@jit(nopython=True)
def C200(r, a, M, theta):
    return (
        -2
        * (a * M) ** 2
        * M
        * r
        * np.cos(theta)
        * np.sin(theta)
        / (r**2 + (a * M) ** 2 * np.cos(theta) ** 2) ** 3
    )


@jit(nopython=True)
def C203(r, a, M, theta):
    return (
        (a * M)
        * M
        * r
        * ((a * M) ** 2 + r**2)
        * np.sin(2 * theta)
        / (r**2 + (a * M) ** 2 * np.cos(theta) ** 2) ** 3
    )


@jit(nopython=True)
def C211(r, a, M, theta):
    return (
        (a * M) ** 2
        * np.cos(theta)
        * np.sin(theta)
        / (((a * M) ** 2 + r * (-2 * M + r)) * (r**2 + (a * M) ** 2 * np.cos(theta) ** 2))
    )


@jit(nopython=True)
def C212(r, a, M, theta):
    return r / (r**2 + (a * M) ** 2 * np.cos(theta) ** 2)


@jit(nopython=True)
def C222(r, a, M, theta):
    return -(
        (a * M) ** 2 * np.cos(theta) * np.sin(theta) / (r**2 + (a * M) ** 2 * np.cos(theta) ** 2)
    )


@jit(nopython=True)
def C233(r, a, M, theta):
    return (
        -(1 / (r**2 + (a * M) ** 2 * np.cos(theta) ** 2) ** 3)
        * np.cos(theta)
        * np.sin(theta)
        * (
            2 * (a * M) ** 2 * r**2 * ((a * M) ** 2 + r**2) * np.cos(theta) ** 2
            + (a * M) ** 4 * ((a * M) ** 2 + r**2) * np.cos(theta) ** 4
            + r
            * (
                (a * M) ** 2 * r**3
                + r**5
                + 4 * (a * M) ** 2 * M * r**2 * np.sin(theta) ** 2
                + 2 * (a * M) ** 4 * M * np.sin(theta) ** 4
                + (a * M) ** 4 * M * np.sin(2 * theta) ** 2
            )
        )
    )


@jit(nopython=True)
def C301(r, a, M, theta):
    return ((a * M) * M * (r**2 - (a * M) ** 2 * np.cos(theta) ** 2)) / (
        ((a * M) ** 2 + r * (r - 2 * M)) * (r**2 + (a * M) ** 2 * np.cos(theta) ** 2) ** 2
    )


@jit(nopython=True)
def C302(r, a, M, theta):
    return (
        -(8 * (a * M) * M * r / np.tan(theta))
        / ((a * M) ** 2 + 2 * r**2 + (a * M) ** 2 * np.cos(2 * theta)) ** 2
    )


@jit(nopython=True)
def C313(r, a, M, theta):
    return (
        (a * M) ** 4 * M
        + 3 * (a * M) ** 4 * r
        - 12 * (a * M) ** 2 * M * r**2
        + 8 * (a * M) ** 2 * r**3
        - 16 * M * r**4
        + 8 * r**5
        + 4 * (a * M) ** 2 * r * ((a * M) ** 2 + r * (2 * r - M)) * np.cos(2 * theta)
        - (a * M) ** 4 * (M - r) * np.cos(4 * theta)
    ) / (
        2
        * ((a * M) ** 2 + r * (r - 2 * M))
        * ((a * M) ** 2 + 2 * r**2 + (a * M) ** 2 * np.cos(2 * theta)) ** 2
    )


@jit(nopython=True)
def C323(r, a, M, theta):
    return (
        3 * (a * M) ** 4
        + 8 * (a * M) ** 2 * M * r
        + 8 * (a * M) ** 2 * r**2
        + 8 * r**4
        + 4 * (a * M) ** 2 * ((a * M) ** 2 + 2 * r * (r - M)) * np.cos(2 * theta)
        + (a * M) ** 4 * np.cos(4 * theta)
    ) / (np.tan(theta) * 2 * ((a * M) ** 2 + 2 * r**2 + (a * M) ** 2 * np.cos(2 * theta)) ** 2)


# ---------------------------------------------------------------------------
# RKF45 (Runge-Kutta-Fehlberg) adaptive-step integrator (Kerr-tracer variant).
#
# Coefficients and step-control logic are untouched from the legacy implementation.
# Kept distinct from `weyl_core.run_kut4_mod` (different hmax and simpler step-halving
# logic) since the two spacetimes are independent.
# ---------------------------------------------------------------------------
@jit(nopython=True)
def run_kut4_mod(F, x, y, h, *args):
    hmax = -0.02
    hmin = -(10**-7)
    tol = 10**-4

    K1 = h * F(x, y, *args)
    K2 = h * F(x + h / 4, y + K1 / 4, *args)
    K3 = h * F(x + 3 / 8 * h, y + 3 / 32 * K1 + 9 / 32 * K2, *args)
    K4 = h * F(x + 12 / 13 * h, y + 1932 / 2197 * K1 - 7200 / 2197 * K2 + 7296 / 2197 * K3, *args)
    K5 = h * F(x + h, y + 439 / 216 * K1 - 8 * K2 + 3680 / 513 * K3 - 845 / 4104 * K4, *args)
    K6 = h * F(
        x + h / 2,
        y - 8 / 27 * K1 + 2 * K2 - 3544 / 2565 * K3 + 1859 / 4104 * K4 - 11 / 40 * K5,
        *args,
    )

    y4 = y + 25 / 216 * K1 + 1408 / 2565 * K3 + 2197 / 4101 * K4 - K5 / 5
    y5 = y + 16 / 135 * K1 + 6656 / 12825 * K3 + 28561 / 56430 * K4 - 9 / 50 * K5 + 2 / 55 * K6

    error = np.linalg.norm(y5 - y4)
    delta = pow(1.0 / 2.0, (1.0 / 4.0)) * pow(tol / error, (1.0 / 4.0))

    while error > tol:
        h = delta * h

        K1 = h * F(x, y, *args)
        K2 = h * F(x + h / 4, y + K1 / 4, *args)
        K3 = h * F(x + 3 / 8 * h, y + 3 / 32 * K1 + 9 / 32 * K2, *args)
        K4 = h * F(
            x + 12 / 13 * h, y + 1932 / 2197 * K1 - 7200 / 2197 * K2 + 7296 / 2197 * K3, *args
        )
        K5 = h * F(x + h, y + 439 / 216 * K1 - 8 * K2 + 3680 / 513 * K3 - 845 / 4104 * K4, *args)
        K6 = h * F(
            x + h / 2,
            y - 8 / 27 * K1 + 2 * K2 - 3544 / 2565 * K3 + 1859 / 4104 * K4 - 11 / 40 * K5,
            *args,
        )

        y4 = y + 25 / 216 * K1 + 1408 / 2565 * K3 + 2197 / 4101 * K4 - K5 / 5
        y5 = y + 16 / 135 * K1 + 6656 / 12825 * K3 + 28561 / 56430 * K4 - 9 / 50 * K5 + 2 / 55 * K6

        error = np.linalg.norm(y5 - y4)
        delta = pow(1.0 / 2.0, (1.0 / 4.0)) * pow(tol / error, (1.0 / 4.0))

    else:
        if np.abs(h) < np.abs(hmin):
            h = hmin
        elif np.abs(h) > np.abs(hmax):
            h = hmax
        else:
            h = delta * h

        x = x + h
        y = y4

    return (h, x, y)


@jit(nopython=True)
def geo(t, z, a, M, alfa, beta, r0, theta0):
    """Kerr geodesic RHS for the 5-state [r, dr, theta, dtheta, phi] system."""
    pphi = Pphi(r0, a, M, theta0, alfa, beta)
    pt = Pt(r0, a, M, theta0, alfa, beta)

    d2Rdt = -(
        C100(z[0], a, M, z[2]) * dt(z[0], a, M, z[2], alfa, beta, pt, pphi) ** 2
        + 2
        * C103(z[0], a, M, z[2])
        * dt(z[0], a, M, z[2], alfa, beta, pt, pphi)
        * dphi(z[0], a, M, z[2], alfa, beta, pt, pphi)
        + C111(z[0], a, M, z[2]) * (z[1]) ** 2
        + 2 * C112(z[0], a, M, z[2]) * z[1] * z[3]
        + C122(z[0], a, M, z[2]) * (z[3]) ** 2
        + C133(z[0], a, M, z[2]) * dphi(z[0], a, M, z[2], alfa, beta, pt, pphi) ** 2
    )

    d2Thedt = -(
        C200(z[0], a, M, z[2]) * (dt(z[0], a, M, z[2], alfa, beta, pt, pphi)) ** 2
        + C203(z[0], a, M, z[2])
        * dt(z[0], a, M, z[2], alfa, beta, pt, pphi)
        * dphi(z[0], a, M, z[2], alfa, beta, pt, pphi)
        + C211(z[0], a, M, z[2]) * (z[1]) ** 2
        + 2 * C212(z[0], a, M, z[2]) * z[1] * z[3]
        + C222(z[0], a, M, z[2]) * (z[3]) ** 2
        + C233(z[0], a, M, z[2]) * dphi(z[0], a, M, z[2], alfa, beta, pt, pphi) ** 2
    )

    dPhidt = gtp_inv(z[0], a, M, z[2]) * pt + gpp_inv(z[0], a, M, z[2]) * pphi

    return np.array([z[1], d2Rdt, z[3], d2Thedt, dPhidt])


@jit(nopython=True)
def func(y, x, h, alfa, beta, a, M, r0, theta0):
    """Trace a single Kerr null geodesic until it falls below the horizon or escapes r > 30.

    Returns the final [r, theta, phi] triple.
    """
    Y = [y]
    yf = [Y[-1][0], Y[-1][2], Y[-1][4]]
    while y[0] > 1.01 * M * (1 + np.sqrt(1 - a**2)):
        (h, x, y) = run_kut4_mod(geo, x, y, h, a, M, alfa, beta, r0, theta0)
        Y.append(y)

        if y[0] > 30:
            yf = [Y[-1][0], Y[-1][2], Y[-1][4]]
            break

        yf = [Y[-1][0], Y[-1][2], Y[-1][4]]

    return yf


@jit(nopython=True)
def func_status(y, x, h, alfa, beta, a, M, r0, theta0):
    """Like `func`, but also returns a status: 0 escaped, 1 captured, 3 numerical failure (NaN)."""
    r_h = 1.01 * M * (1 + np.sqrt(1 - a**2))
    while y[0] > r_h:
        (h, x, y) = run_kut4_mod(geo, x, y, h, a, M, alfa, beta, r0, theta0)
        if np.isnan(y[0]) or np.isnan(y[2]) or np.isnan(y[4]):
            return y[0], y[2], y[4], 3
        if y[0] > 30:
            return y[0], y[2], y[4], 0
    return y[0], y[2], y[4], 1


@jit(nopython=True, parallel=True)
def f_paral(r0, theta0, a, M, alfa, beta):
    """Multicore driver: trace one ray per (alfa, beta) grid point.

    Returns (r, theta, phi, status) matrices of shape (len(alfa), len(beta)).
    """
    Mr = np.zeros((len(alfa), len(beta)))
    Mthet = np.zeros((len(alfa), len(beta)))
    Mphi = np.zeros((len(alfa), len(beta)))
    Mstat = np.zeros((len(alfa), len(beta)), dtype=np.int64)

    for i in prange(len(alfa)):
        for j in range(len(beta)):
            y = np.array(
                [
                    r0,
                    dr(r0, a, M, theta0, alfa[i], beta[j]),
                    theta0,
                    dthe(r0, a, M, theta0, alfa[i]),
                    0.0,
                ]
            )
            (Mr[i, j], Mthet[i, j], Mphi[i, j], Mstat[i, j]) = func_status(
                y, 300.0, -0.02, alfa[i], beta[j], a, M, r0, theta0
            )

    return (Mr, Mthet, Mphi, Mstat)


# ---------------------------------------------------------------------------
# Analytic photon-ring overlay helpers (equatorial, spherical photon orbits)
# ---------------------------------------------------------------------------
def r1(a, M):
    return 2 * (1 + np.cos((2 / 3) * np.arccos(-(np.abs(a * M) / M))))


def r2(a, M):
    return 2 * (1 + np.cos((2 / 3) * np.arccos(np.abs(a * M) / M)))


def lamb(r, a, M):
    return -((r / M) ** 3 - 3 * (r / M) ** 2 + (a * M) ** 2 * r / M + (a * M) ** 2) / (
        a * M * (r / M - 1)
    )


def eta(r, a, M):
    return (
        -((r / M) ** 3)
        * ((r / M) ** 3 - 6 * (r / M) ** 2 + 9 * r / M - 4 * (a * M) ** 2)
        / ((a * M) ** 2 * (r / M - 1) ** 2)
    )


def yplus(r, a, M, theta, r0):
    return np.sqrt(gpp(r0, a, M, theta)) * np.arcsin(
        (1 / (zeta(r0, a, M, theta) - lamb(r, a, M) * gamma(r0, a, M, theta)))
        * np.sqrt(eta(r, a, M))
        / np.sqrt(r0**2 + (a * M) ** 2 * np.cos(theta) ** 2)
    )


def yminus(r, a, M, theta, r0):
    return np.sqrt(gpp(r0, a, M, theta)) * np.arcsin(
        (-1 / (zeta(r0, a, M, theta) - lamb(r, a, M) * gamma(r0, a, M, theta)))
        * np.sqrt(eta(r, a, M))
        / np.sqrt(r0**2 + (a * M) ** 2 * np.cos(theta) ** 2)
    )


def x(r, a, M, theta, r0):
    return -np.sqrt(gpp(r0, a, M, theta)) * np.arctan(
        lamb(r, a, M)
        * np.sqrt((r0**2 + (a * M) ** 2 * np.cos(theta) ** 2) * delta(r0, a, M))
        / (
            np.sqrt(gpp(r0, a, M, theta))
            * np.sqrt(
                r0**4
                + ((a * M) ** 2 - eta(r, a, M) - lamb(r, a, M) ** 2) * r0**2
                + 2 * M * r0 * (eta(r, a, M) + ((a * M) - lamb(r, a, M)) ** 2)
                - eta(r, a, M) * (a * M) ** 2
            )
        )
    )


@jit(nopython=True, parallel=True)
def retrace_points(r0, theta0, a, M, alfa, beta, ii, jj):
    """Re-trace the k pixels `(alfa[ii[k]], beta[jj[k]])` in parallel; same rays as `f_paral`."""
    k = len(ii)
    Mr = np.zeros(k)
    Mthet = np.zeros(k)
    Mphi = np.zeros(k)
    Mstat = np.zeros(k, dtype=np.int64)

    for p in prange(k):
        i = ii[p]
        j = jj[p]
        y = np.array(
            [
                r0,
                dr(r0, a, M, theta0, alfa[i], beta[j]),
                theta0,
                dthe(r0, a, M, theta0, alfa[i]),
                0.0,
            ]
        )
        (Mr[p], Mthet[p], Mphi[p], Mstat[p]) = func_status(
            y, 300.0, -0.02, alfa[i], beta[j], a, M, r0, theta0
        )

    return (Mr, Mthet, Mphi, Mstat)
