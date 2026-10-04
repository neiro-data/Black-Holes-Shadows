"""Weyl (Schwarzschild + optional Morgan-Morgan disk) null-geodesic tracer on the Apple GPU.

MLX Metal kernel, float32, one thread per pixel. Port of `weyl_core.py` (`nu`, `lamb`, `geo`,
RKF45 `run_kut4_mod`, `func`). Differences from the CPU float64 path, forced by float32:

- `derNU` is not a `1e-6` central difference (useless in float32). The Schwarzschild part has
  analytic derivatives; the disk part (`nuD`, closed form) uses a float32 central difference with
  a scaled step `h = HDER_GPU * max(1, |x|)`. For `0 < |z| < h` the z derivative is a one-sided
  second-order difference (the central stencil would straddle the kink of `nuD` at z=0).
  `dlamb2` is built from these derivatives, as on CPU.
- `nuD` is evaluated in an algebraically identical, cancellation-free form
  (`w + sqrt(w^2 + 4 a^2 z^2) = 4 a^2 z^2 / (sqrt(...) - w)` for `w < 0`).
- `lamb` is the same bilinear lookup on the (float32) `Mat_nu` table, passed as a 2-D input.
- The step-size controller uses `hmin = -1e-5`, a retry cap, a per-ray step cap and a clean
  retry floor, whereas the CPU retry branch has legacy quirks (it compares `h < hmin` on negative
  steps and may loop past 500 retries). Non-finite state or an unmet tolerance -> status 3.

The kernel returns the same `[rho, z]` sentinels as `weyl_core.func` (escape -> `[rho, z]`,
capture -> `[0.001, 0.001]`, disk hit -> `[rho, z + 50]`), so `classify_shadow_disk` is shared with
the CPU path. Status 3 pixels are re-traced on the CPU (float64) by the runner.
"""

from __future__ import annotations

import numpy as np

from bh_shadows.core import weyl_core as wc
from bh_shadows.gpu import backend as gb

HDER_GPU = 1e-3
HMIN_GPU = -1e-5
MAX_STEPS = 200_000
MAX_RETRIES = 50

_HEADER = f"""
#define TOL 1e-4f
#define HMAX (-0.04f)
#define HDER_GPU {HDER_GPU!r}f
#define SQ2 1.41421356237f
#define PI_F 3.14159265359f

struct P {{
  float M, MD, b, rho0, z0, pt, pphi, gpp0, grr0, gzz0;
  int vis, ni, nj;
  const device float* lam;
}};

inline float nuSch_(float rho, float z, float M) {{
  float d1 = sqrt(rho*rho + (z - M)*(z - M)), d2 = sqrt(rho*rho + (z + M)*(z + M));
  float S = d1 + d2;
  return log((S - 2*M)/(S + 2*M));
}}

// Analytic derivatives of nuSch.
inline void dnuSch_(float rho, float z, float M, thread float& nr, thread float& nz) {{
  float d1 = sqrt(rho*rho + (z - M)*(z - M)), d2 = sqrt(rho*rho + (z + M)*(z + M));
  float S = d1 + d2;
  float fac = 1/(S - 2*M) - 1/(S + 2*M);
  nr = fac*(rho/d1 + rho/d2);
  nz = fac*((z - M)/d1 + (z + M)/d2);
}}

// Morgan-Morgan disk potential (closed form, both xi2 branches).
inline float nuD_(float R, float z, float M, float a) {{
  float R2 = R*R, z2 = z*z, a2 = a*a, rr = R2 + z2;
  float w = a2 - rr;
  float S = sqrt(w*w + 4*a2*z2);
  float wS = (w < 0) ? 4*a2*z2/(S - w) : (w + S);
  float t1 = M*sqrt(fabs(wS))*(-3*a2 + rr + 3*S)/(SQ2*PI_F*rr*rr);
  float A = (wS == 0) ? PI_F/2 : atan(SQ2*sqrt(rr/fabs(wS)));
  if (!isfinite(A)) A = PI_F/2;
  float t2 = 2*M*(-(a2*(R2 - 2*z2)) + 2*rr*rr)*A/(PI_F*rr*rr*sqrt(rr));
  return t1 - t2;
}}

// nu = nuSch + nuD and its (rho, z) derivatives.
inline float nu_val(float rho, float z, thread const P& p) {{
  float v = nuSch_(rho, z, p.M);
  if (p.MD != 0) v += nuD_(rho, z, p.MD, p.b);
  return v;
}}

inline void nu_all(float rho, float z, thread const P& p,
                   thread float& nu, thread float& nr, thread float& nz) {{
  nu = nuSch_(rho, z, p.M);
  dnuSch_(rho, z, p.M, nr, nz);
  if (p.MD != 0) {{
    float hr = HDER_GPU*max(1.f, fabs(rho)), hz = HDER_GPU*max(1.f, fabs(z));
    nu += nuD_(rho, z, p.MD, p.b);
    nr += 0.5f*(nuD_(rho + hr, z, p.MD, p.b) - nuD_(rho - hr, z, p.MD, p.b))/hr;
    if (z != 0.f && fabs(z) < hz) {{
      // nuD is even in z with a kink at z=0: a central stencil would straddle it, so use a
      // 2nd-order one-sided difference on the side of z.
      float s = z > 0.f ? 1.f : -1.f;
      nz += s*(-3.f*nuD_(rho, z, p.MD, p.b) + 4.f*nuD_(rho, z + s*hz, p.MD, p.b)
               - nuD_(rho, z + 2.f*s*hz, p.MD, p.b))/(2.f*hz);
    }} else {{
      nz += 0.5f*(nuD_(rho, z + hz, p.MD, p.b) - nuD_(rho, z - hz, p.MD, p.b))/hz;
    }}
  }}
}}

// Bilinear lookup of the lambda table (port of `weyl_core.lamb`, incl. the clamp branch).
inline float lamb_(float rho, float z, thread const P& p) {{
  float fi = (40.f - z)/40.f*(p.ni - 1)/2;
  float fj = (40.f - rho)/40.f*(p.nj - 1);
  int i0 = (int)floor(fi), j0 = (int)floor(fj);
  if (fabs(fj) > p.nj - 1) {{
    fi = (float)(p.ni/2) + 0.5f; fj = p.nj - 0.5f;
    i0 = (int)floor(fi); j0 = (int)floor(fj);
  }}
  i0 = clamp(i0, 0, p.ni - 2); j0 = clamp(j0, 0, p.nj - 2);
  float I = fi - i0, J = fj - j0;
  float f00 = p.lam[i0*p.nj + j0], f01 = p.lam[i0*p.nj + j0 + 1];
  float f10 = p.lam[(i0 + 1)*p.nj + j0], f11 = p.lam[(i0 + 1)*p.nj + j0 + 1];
  return (f00 + (f01 - f00)*J) + (f10 - f00 + (f11 + f00 - f01 - f10)*J)*I;
}}

inline float sgn(float x) {{ return (x > 0) - (x < 0); }}

// sqrt(gpp(sqrt(rho^2 + z^2), 0)): the escape-radius proxy used by `func`.
inline float esc_r(float rho, float z, thread const P& p) {{
  float R = sqrt(rho*rho + z*z);
  return R*exp(-0.5f*nu_val(R, 0.f, p));
}}

// RHS of the 4-state system [rho, drho, z, dz].
inline void geo(thread const float* y, thread const P& p, thread float* out) {{
  float rho = y[0], z = y[2];
  float nu, nr, nz;
  nu_all(rho, z, p, nu, nr, nz);
  float lam = lamb_(rho, z, p);
  float dl0 = 0.5f*rho*(nr*nr - nz*nz), dl1 = rho*nr*nz;
  float dt = -p.pt*exp(-nu);
  float dph = p.pphi/(rho*rho*exp(-nu));
  float e2 = exp(2*nu - lam), em = exp(-lam);
  out[0] = y[1];
  out[1] = -(0.5f*e2*nr*dt*dt + 0.5f*(dl0 - nr)*y[1]*y[1] + (dl1 - nz)*y[1]*y[3]
             - 0.5f*(dl0 - nr)*y[3]*y[3] + 0.5f*em*rho*(-2 + rho*nr)*dph*dph);
  out[2] = y[3];
  out[3] = -(0.5f*e2*nz*dt*dt - 0.5f*(dl1 - nz)*y[1]*y[1] + (dl0 - nr)*y[1]*y[3]
             + 0.5f*(dl1 - nz)*y[3]*y[3] + 0.5f*em*rho*rho*nz*dph*dph);
}}

// One RKF45 attempt of size h: fills y4 and returns ||y5 - y4||.
inline float rkf(thread const float* y, float h, thread const P& p, thread float* y4) {{
  float K1[4], K2[4], K3[4], K4[4], K5[4], K6[4], t[4], f[4];
  geo(y, p, f); for (int i = 0; i < 4; i++) K1[i] = h*f[i];
  for (int i = 0; i < 4; i++) t[i] = y[i] + K1[i]/4;
  geo(t, p, f); for (int i = 0; i < 4; i++) K2[i] = h*f[i];
  for (int i = 0; i < 4; i++) t[i] = y[i] + 3.f/32*K1[i] + 9.f/32*K2[i];
  geo(t, p, f); for (int i = 0; i < 4; i++) K3[i] = h*f[i];
  for (int i = 0; i < 4; i++)
    t[i] = y[i] + 1932.f/2197*K1[i] - 7200.f/2197*K2[i] + 7296.f/2197*K3[i];
  geo(t, p, f); for (int i = 0; i < 4; i++) K4[i] = h*f[i];
  for (int i = 0; i < 4; i++)
    t[i] = y[i] + 439.f/216*K1[i] - 8*K2[i] + 3680.f/513*K3[i] - 845.f/4104*K4[i];
  geo(t, p, f); for (int i = 0; i < 4; i++) K5[i] = h*f[i];
  for (int i = 0; i < 4; i++)
    t[i] = y[i] - 8.f/27*K1[i] + 2*K2[i] - 3544.f/2565*K3[i] + 1859.f/4104*K4[i]
           - 11.f/40*K5[i];
  geo(t, p, f); for (int i = 0; i < 4; i++) K6[i] = h*f[i];
  float e2 = 0;
  for (int i = 0; i < 4; i++) {{
    y4[i] = y[i] + 25.f/216*K1[i] + 1408.f/2565*K3[i] + 2197.f/4101*K4[i] - K5[i]/5;
    float y5 = y[i] + 16.f/135*K1[i] + 6656.f/12825*K3[i] + 28561.f/56430*K4[i]
               - 9.f/50*K5[i] + 2.f/55*K6[i];
    e2 += (y5 - y4[i])*(y5 - y4[i]);
  }}
  return sqrt(e2);
}}

// Fill P from the prm vector: [rho0, z0, M, MD, b, pt, gpp0, vis, grr0, gzz0].
inline void load_p(const device float* prm, int ni, int nj, const device float* lam,
                   thread P& p) {{
  p.rho0 = prm[0]; p.z0 = prm[1]; p.M = prm[2]; p.MD = prm[3]; p.b = prm[4];
  p.pt = prm[5]; p.gpp0 = prm[6]; p.vis = (int)prm[7]; p.grr0 = prm[8]; p.gzz0 = prm[9];
  p.ni = ni; p.nj = nj; p.lam = lam;
}}

// Pixel initial state; sets p.pphi (observer momenta are constants along the ray).
inline void init_ray(float alpha, float beta, thread P& p, thread float* y) {{
  p.pphi = sqrt(p.gpp0)*sin(beta)*cos(alpha);
  y[0] = p.rho0;
  y[1] = cos(alpha)*cos(beta)/sqrt(p.grr0);
  y[2] = p.z0;
  y[3] = sin(alpha)/sqrt(p.gzz0);
}}
"""

_TRACE = f"""
uint idx = thread_position_in_grid.x;
uint nb = (uint)beta_shape[0];
uint total = (uint)alpha_shape[0]*nb;
if (idx >= total) return;
P p;
load_p(prm, (int)lam_shape[0], (int)lam_shape[1], lam, p);
float y[4];
init_ray(alpha[idx / nb], beta[idx % nb], p, y);
float hmin = {HMIN_GPU!r}f, h = -0.02f;
float orho = y[0], oz = y[2];
int st = 0;
bool done = nu_val(y[0], y[2], p) <= -3.f || esc_r(y[0], y[2], p) >= 30.f;
float zprev = y[2];
for (int step = 0; !done; step++) {{
  if (step >= {MAX_STEPS}) {{ st = 3; break; }}
  float y4[4];
  float err = rkf(y, h, p, y4);
  float delta = pow(0.5f, 0.25f)*pow(TOL/err, 0.25f);
  int tries = 0;
  while (err > TOL && tries < {MAX_RETRIES}) {{
    h = delta*h;
    bool floored = fabs(h) < fabs(hmin);
    if (floored) h = hmin;
    err = rkf(y, h, p, y4);
    delta = pow(0.5f, 0.25f)*pow(TOL/err, 0.25f);
    tries++;
    if (floored) break;
  }}
  if (err > TOL) {{ st = 3; break; }}
  if (tries == 0) {{
    float nh = delta*h;
    if (fabs(nh) < fabs(hmin)) h = hmin;
    else if (fabs(nh) > fabs(HMAX)) h = HMAX;
    else h = nh;
  }}
  for (int i = 0; i < 4; i++) y[i] = y4[i];
  orho = y[0]; oz = y[2];
  if (!isfinite(y[0]) || !isfinite(y[1]) || !isfinite(y[2]) || !isfinite(y[3])) {{ st = 3; break; }}
  if (esc_r(y[0], y[2], p) >= 30.f) break;
  else if (nu_val(y[0], y[2], p) <= -3.f) {{ orho = 0.001f; oz = 0.001f; break; }}
  else if (p.vis != 0 && y[0] > p.b && sgn(y[2]) == -sgn(zprev)) {{ oz = y[2] + 50.f; break; }}
  zprev = y[2];
}}
out_rho[idx] = orho; out_z[idx] = oz; out_st[idx] = st;
"""

# Debug kernel: nu, lamb, d nu/d rho, d nu/d z at given (rho, z) points.
_PROBE = """
uint idx = thread_position_in_grid.x;
if (idx >= (uint)pts_shape[0]/2) return;
P p;
p.M = prm[0]; p.MD = prm[1]; p.b = prm[2];
p.ni = (int)lam_shape[0]; p.nj = (int)lam_shape[1]; p.lam = lam;
float rho = pts[idx*2], z = pts[idx*2 + 1];
float nu, nr, nz;
nu_all(rho, z, p, nu, nr, nz);
out_nu[idx] = nu; out_lam[idx] = lamb_(rho, z, p); out_nr[idx] = nr; out_nz[idx] = nz;
"""

# Debug kernel: the geodesic RHS at given states (validates `geo` against `weyl_core.geo`).
_GEO = """
uint idx = thread_position_in_grid.x;
if (idx >= (uint)zin_shape[0]/4) return;
P p;
load_p(prm, (int)lam_shape[0], (int)lam_shape[1], lam, p);
float y0[4];
init_ray(prm[10], prm[11], p, y0);
float z[4], o[4];
for (int i = 0; i < 4; i++) z[i] = zin[idx*4 + i];
geo(z, p, o);
for (int i = 0; i < 4; i++) out[idx*4 + i] = o[i];
"""


def _check_lam(lam) -> np.ndarray:
    """Make sure that the lambda table is loaded.

    Parameters
    ----------
    lam : array_like
        Lambda table.

    Returns
    -------
    np.ndarray
        The table as an array.

    Raises
    ------
    ValueError
        If the table is not 2-D or is the (4, 4) placeholder.
    """
    lam = np.asarray(lam)
    if lam.ndim != 2 or lam.shape == (4, 4):
        raise ValueError(
            f"lambda matrix not loaded (got shape {lam.shape}); call load_matrix first"
        )
    return lam


def _observer_prm(rho0, z0, M, MD, b, disk_visible) -> list[float]:
    """Calculate the observer constants on the host in float64.

    Parameters
    ----------
    rho0, z0 : float
        Observer position.
    M, MD, b : float
        Black-hole mass, disk mass and disk inner radius.
    disk_visible : bool
        Stop rays at the disk.

    Returns
    -------
    list of float
        `[rho0, z0, M, MD, b, pt, gpp0, vis, grr0, gzz0]`.
    """
    gtt0 = wc.gtt(rho0, z0, M, MD, b)
    return [
        rho0,
        z0,
        M,
        MD,
        b,
        -1.0 / np.sqrt(-1.0 / gtt0),
        wc.gpp(rho0, z0, M, MD, b),
        float(bool(disk_visible)),
        wc.grr(rho0, z0, M, MD, b),
        wc.gzz(rho0, z0, M, MD, b),
    ]


def f_paral_gpu(rho0, z0, M, MD, b, alpha, beta, disk_visible, lam):
    """Trace one ray for each pixel on the GPU. GPU version of `weyl_core.f_paral`.

    Parameters
    ----------
    rho0, z0 : float
        Observer position.
    M, MD, b : float
        Black-hole mass, disk mass and disk inner radius.
    alpha, beta : np.ndarray
        Pixel angles.
    disk_visible : bool
        Stop rays at the disk.
    lam : np.ndarray
        Lambda table (`weyl_core.Mat_nu`).

    Returns
    -------
    Mat, Mz : np.ndarray
        End rho and end z, of shape (len(alpha), len(beta)). The sentinels are the same as in
        `weyl_core.func`.
    status : np.ndarray
        0 done, 3 failed in float32 (trace again on the CPU).
    """
    lam = _check_lam(lam)
    alpha, beta = np.asarray(alpha), np.asarray(beta)
    na, nb = len(alpha), len(beta)
    res = gb.launch(
        "weyl_trace",
        _HEADER,
        _TRACE,
        {
            "alpha": alpha,
            "beta": beta,
            "prm": np.array(_observer_prm(rho0, z0, M, MD, b, disk_visible)),
            "lam": lam,
        },
        {"out_rho": np.float32, "out_z": np.float32, "out_st": np.int32},
        na * nb,
    )
    shape = (na, nb)
    return (
        res["out_rho"].astype(np.float64).reshape(shape),
        res["out_z"].astype(np.float64).reshape(shape),
        res["out_st"].astype(np.int64).reshape(shape),
    )


def probe_gpu(points, M, MD, b, lam):
    """Evaluate the potentials on the GPU at given points. For tests.

    Parameters
    ----------
    points : array_like
        Points `(rho, z)` of shape (k, 2).
    M, MD, b : float
        Black-hole mass, disk mass and disk inner radius.
    lam : np.ndarray
        Lambda table.

    Returns
    -------
    tuple of np.ndarray
        `nu`, `lamb`, `d nu / d rho` and `d nu / d z` at each point.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    k = len(pts)
    res = gb.launch(
        "weyl_probe",
        _HEADER,
        _PROBE,
        {"pts": pts.ravel(), "prm": np.array([M, MD, b]), "lam": _check_lam(lam)},
        {"out_nu": np.float32, "out_lam": np.float32, "out_nr": np.float32, "out_nz": np.float32},
        k,
    )
    return tuple(res[n].astype(np.float64) for n in ("out_nu", "out_lam", "out_nr", "out_nz"))


def geo_gpu(states, rho0, z0, M, MD, b, alpha, beta, lam):
    """Evaluate the geodesic right-hand side on the GPU. For tests against `weyl_core.geo`.

    Parameters
    ----------
    states : array_like
        States `[rho, drho, z, dz]` of shape (k, 4).
    rho0, z0 : float
        Observer position.
    M, MD, b : float
        Black-hole mass, disk mass and disk inner radius.
    alpha, beta : float
        Pixel angles of the ray.
    lam : np.ndarray
        Lambda table.

    Returns
    -------
    np.ndarray
        Derivatives of shape (k, 4).
    """
    states = np.asarray(states, dtype=np.float32).reshape(-1, 4)
    prm = np.array(_observer_prm(rho0, z0, M, MD, b, False) + [alpha, beta])
    res = gb.launch(
        "weyl_geo",
        _HEADER,
        _GEO,
        {"zin": states.ravel(), "prm": prm, "lam": _check_lam(lam)},
        {"out": np.float32},
        len(states),
        out_len=states.size,
    )
    return res["out"].astype(np.float64).reshape(-1, 4)
