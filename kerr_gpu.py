"""Kerr null-geodesic tracer on the Apple GPU (MLX Metal kernel, float32, one thread per pixel).

Port of `kerr_core.py` (metric, Christoffels, `geo`, RKF45, `func_status`). Differences from the
CPU float64 path, forced by float32: `hmin = -1e-5`, a retry cap, a per-ray step cap, and
non-finite state -> status 3. Status 3 pixels are re-traced on the CPU by the runner.
"""

from __future__ import annotations

import numpy as np

import gpu_backend as gb

HMIN_GPU = -1e-5
MAX_STEPS = 200_000
MAX_RETRIES = 50

_HEADER = """
#define TOL 1e-4f
#define HMAX (-0.02f)

struct P { float a, M, r0, th0, pt, pphi; };

inline float den(float r, float am, float th) { return r*r + am*am*cos(th)*cos(th); }
inline float D2(float r, float am, float th) { return am*am + 2*r*r + am*am*cos(2*th); }
inline float Dl(float r, float am, float M) { return am*am + r*(r - 2*M); }

inline float gtt_(float r, float am, float M, float th) { return -(1 - 2*M*r/den(r, am, th)); }
inline float gtp_(float r, float am, float M, float th) {
  float s = sin(th); return -2*M*am*r*s*s/den(r, am, th); }
inline float gpp_(float r, float am, float M, float th) {
  float s = sin(th);
  return s*s*(r*r + am*am + 2*M*r*am*am*s*s/den(r, am, th)); }
inline float grr_(float r, float am, float M, float th) { return den(r, am, th)/Dl(r, am, M); }

inline float gpp_inv(float r, float am, float M, float th) {
  float s = sin(th);
  return (am*am + 2*r*(r - 2*M) + am*am*cos(2*th))*(1/(s*s))/(Dl(r, am, M)*D2(r, am, th)); }
inline float gtt_inv(float r, float am, float M, float th) {
  return -(am*am*am*am + 2*r*r*r*r + am*am*r*(2*M + 3*r)
           + am*am*Dl(r, am, M)*cos(2*th))/(Dl(r, am, M)*D2(r, am, th)); }
inline float gtp_inv(float r, float am, float M, float th) {
  return -4*am*M*r/(Dl(r, am, M)*D2(r, am, th)); }

inline float C001(float r, float am, float M, float th) {
  float d2 = D2(r, am, th);
  return -2*M*(am*am + r*r)*(am*am - 2*r*r + am*am*cos(2*th))/(Dl(r, am, M)*d2*d2); }
inline float C002(float r, float am, float M, float th) {
  float d2 = D2(r, am, th);
  return -(4*am*am*M*r*sin(2*th))/(d2*d2); }
inline float C013(float r, float am, float M, float th) {
  float d2 = D2(r, am, th), s = sin(th), a2 = am*am, r2 = r*r;
  return 2*am*M*(a2*a2 - 3*a2*r2 - 6*r2*r2 + a2*(a2 - r2)*cos(2*th))*s*s/(Dl(r, am, M)*d2*d2); }
inline float C023(float r, float am, float M, float th) {
  float d2 = D2(r, am, th), s = sin(th);
  return 8*am*am*am*M*r*cos(th)*s*s*s/(d2*d2); }

inline float C100(float r, float am, float M, float th) {
  float d = den(r, am, th);
  return M*(-(am*am) + (2*M - r)*r)*(am*am*cos(th)*cos(th) - r*r)/(d*d*d); }
inline float C103(float r, float am, float M, float th) {
  float d = den(r, am, th), s = sin(th);
  return am*M*Dl(r, am, M)*(-(r*r) + am*am*cos(th)*cos(th))*s*s/(d*d*d); }
inline float C111(float r, float am, float M, float th) {
  float c = cos(th);
  return (r*(am*am - M*r) + am*am*(M - r)*c*c)/(Dl(r, am, M)*den(r, am, th)); }
inline float C112(float r, float am, float M, float th) {
  return -(am*am*cos(th)*sin(th)/den(r, am, th)); }
inline float C122(float r, float am, float M, float th) {
  return -r*Dl(r, am, M)/den(r, am, th); }
inline float C133(float r, float am, float M, float th) {
  float d = den(r, am, th), s = sin(th), c = cos(th), a2 = am*am, a4 = a2*a2;
  return -(Dl(r, am, M)*s*s*(r*r*r*r*r + a4*r*c*c*c*c - a2*M*r*r*s*s
           + c*c*(2*a2*r*r*r + a4*M*s*s)))/(d*d*d); }

inline float C200(float r, float am, float M, float th) {
  float d = den(r, am, th);
  return -2*am*am*M*r*cos(th)*sin(th)/(d*d*d); }
inline float C203(float r, float am, float M, float th) {
  float d = den(r, am, th);
  return am*M*r*(am*am + r*r)*sin(2*th)/(d*d*d); }
inline float C211(float r, float am, float M, float th) {
  return am*am*cos(th)*sin(th)/(Dl(r, am, M)*den(r, am, th)); }
inline float C212(float r, float am, float M, float th) { return r/den(r, am, th); }
inline float C222(float r, float am, float M, float th) {
  return -(am*am*cos(th)*sin(th)/den(r, am, th)); }
inline float C233(float r, float am, float M, float th) {
  float d = den(r, am, th), s = sin(th), c = cos(th), a2 = am*am, a4 = a2*a2, r2 = r*r;
  float s2t = sin(2*th);
  return -(1/(d*d*d))*c*s*(2*a2*r2*(a2 + r2)*c*c + a4*(a2 + r2)*c*c*c*c
           + r*(a2*r2*r + r2*r2*r + 4*a2*M*r2*s*s + 2*a4*M*s*s*s*s + a4*M*s2t*s2t)); }

inline float C301(float r, float am, float M, float th) {
  float d = den(r, am, th);
  return am*M*(r*r - am*am*cos(th)*cos(th))/(Dl(r, am, M)*d*d); }
inline float C302(float r, float am, float M, float th) {
  float d2 = D2(r, am, th);
  return -(8*am*M*r/tan(th))/(d2*d2); }
inline float C313(float r, float am, float M, float th) {
  float d2 = D2(r, am, th), a2 = am*am, a4 = a2*a2, r2 = r*r;
  return (a4*M + 3*a4*r - 12*a2*M*r2 + 8*a2*r2*r - 16*M*r2*r2 + 8*r2*r2*r
          + 4*a2*r*(a2 + r*(2*r - M))*cos(2*th) - a4*(M - r)*cos(4*th))
         /(2*Dl(r, am, M)*d2*d2); }
inline float C323(float r, float am, float M, float th) {
  float d2 = D2(r, am, th), a2 = am*am, a4 = a2*a2, r2 = r*r;
  return (3*a4 + 8*a2*M*r + 8*a2*r2 + 8*r2*r2
          + 4*a2*(a2 + 2*r*(r - M))*cos(2*th) + a4*cos(4*th))/(tan(th)*2*d2*d2); }

// Observer-dependent momenta (constant along a ray); alfa/beta are the pixel angles.
inline float gamma_(float r, float am, float M, float th) {
  float gp = gpp_(r, am, M, th), gt = gtp_(r, am, M, th);
  return (-gt/gp)*sqrt(gp/(gt*gt - gtt_(r, am, M, th)*gp)); }
inline float zeta_(float r, float am, float M, float th) {
  float gp = gpp_(r, am, M, th), gt = gtp_(r, am, M, th);
  return sqrt(gp/(gt*gt - gtt_(r, am, M, th)*gp)); }

// RHS of the 5-state system [r, dr, theta, dtheta, phi].
inline void geo(thread const float* z, thread const P& p, thread float* out) {
  float am = p.a*p.M, M = p.M, r = z[0], th = z[2];
  float dt = gtt_inv(r, am, M, th)*p.pt + gtp_inv(r, am, M, th)*p.pphi;
  float dph = gpp_inv(r, am, M, th)*p.pphi + gtp_inv(r, am, M, th)*p.pt;
  out[0] = z[1];
  out[1] = -(C100(r, am, M, th)*dt*dt + 2*C103(r, am, M, th)*dt*dph
             + C111(r, am, M, th)*z[1]*z[1] + 2*C112(r, am, M, th)*z[1]*z[3]
             + C122(r, am, M, th)*z[3]*z[3] + C133(r, am, M, th)*dph*dph);
  out[2] = z[3];
  out[3] = -(C200(r, am, M, th)*dt*dt + C203(r, am, M, th)*dt*dph
             + C211(r, am, M, th)*z[1]*z[1] + 2*C212(r, am, M, th)*z[1]*z[3]
             + C222(r, am, M, th)*z[3]*z[3] + C233(r, am, M, th)*dph*dph);
  out[4] = gtp_inv(r, am, M, th)*p.pt + gpp_inv(r, am, M, th)*p.pphi;
}

inline void stage(thread const float* y, thread const float* k, float c, thread float* o) {
  for (int i = 0; i < 5; i++) o[i] = y[i] + c*k[i];
}

// One RKF45 attempt of size h: fills y4 and returns ||y5 - y4||.
inline float rkf(thread const float* y, float h, thread const P& p, thread float* y4) {
  float K1[5], K2[5], K3[5], K4[5], K5[5], K6[5], t[5], f[5];
  geo(y, p, f); for (int i = 0; i < 5; i++) K1[i] = h*f[i];
  for (int i = 0; i < 5; i++) t[i] = y[i] + K1[i]/4;
  geo(t, p, f); for (int i = 0; i < 5; i++) K2[i] = h*f[i];
  for (int i = 0; i < 5; i++) t[i] = y[i] + 3.f/32*K1[i] + 9.f/32*K2[i];
  geo(t, p, f); for (int i = 0; i < 5; i++) K3[i] = h*f[i];
  for (int i = 0; i < 5; i++)
    t[i] = y[i] + 1932.f/2197*K1[i] - 7200.f/2197*K2[i] + 7296.f/2197*K3[i];
  geo(t, p, f); for (int i = 0; i < 5; i++) K4[i] = h*f[i];
  for (int i = 0; i < 5; i++)
    t[i] = y[i] + 439.f/216*K1[i] - 8*K2[i] + 3680.f/513*K3[i] - 845.f/4104*K4[i];
  geo(t, p, f); for (int i = 0; i < 5; i++) K5[i] = h*f[i];
  for (int i = 0; i < 5; i++)
    t[i] = y[i] - 8.f/27*K1[i] + 2*K2[i] - 3544.f/2565*K3[i] + 1859.f/4104*K4[i]
           - 11.f/40*K5[i];
  geo(t, p, f); for (int i = 0; i < 5; i++) K6[i] = h*f[i];
  float e2 = 0;
  for (int i = 0; i < 5; i++) {
    y4[i] = y[i] + 25.f/216*K1[i] + 1408.f/2565*K3[i] + 2197.f/4101*K4[i] - K5[i]/5;
    float y5 = y[i] + 16.f/135*K1[i] + 6656.f/12825*K3[i] + 28561.f/56430*K4[i]
               - 9.f/50*K5[i] + 2.f/55*K6[i];
    e2 += (y5 - y4[i])*(y5 - y4[i]);
  }
  return sqrt(e2);
}

// Pixel initial state and observer momenta.
inline void init_ray(float alfa, float beta, thread P& p, thread float* y) {
  float am = p.a*p.M, M = p.M, r0 = p.r0, th0 = p.th0;
  float gp = gpp_(r0, am, M, th0);
  p.pphi = sqrt(gp)*sin(beta)*cos(alfa);
  p.pt = -(1 + gamma_(r0, am, M, th0)*sqrt(gp)*sin(beta)*cos(alfa))/zeta_(r0, am, M, th0);
  float g_th = den(r0, am, th0);
  y[0] = r0;
  y[1] = sqrt(grr_(r0, am, M, th0))*cos(beta)*cos(alfa)*Dl(r0, am, M)/den(r0, am, th0);
  y[2] = th0;
  y[3] = sqrt(g_th)*sin(alfa)/den(r0, am, th0);
  y[4] = 0;
}
"""

_TRACE = f"""
uint idx = thread_position_in_grid.x;
uint nb = (uint)beta_shape[0];
uint total = (uint)alfa_shape[0]*nb;
if (idx >= total) return;
P p;
p.a = prm[0]; p.M = prm[1]; p.r0 = prm[2]; p.th0 = prm[3];
float y[5];
init_ray(alfa[idx / nb], beta[idx % nb], p, y);
float r_h = 1.01f*p.M*(1 + sqrt(1 - p.a*p.a));
float h = -0.02f, hmin = {HMIN_GPU!r}f;
int st = 1;
for (int step = 0; y[0] > r_h; step++) {{
  if (step >= {MAX_STEPS}) {{ st = 3; break; }}
  float y4[5];
  float err = rkf(y, h, p, y4);
  float delta = pow(0.5f, 0.25f)*pow(TOL/err, 0.25f);
  int tries = 0;
  while (err > TOL && tries < {MAX_RETRIES}) {{
    h = delta*h;
    err = rkf(y, h, p, y4);
    delta = pow(0.5f, 0.25f)*pow(TOL/err, 0.25f);
    tries++;
  }}
  if (err > TOL) {{ st = 3; break; }}
  if (fabs(h) < fabs(hmin)) h = hmin;
  else if (fabs(h) > fabs(HMAX)) h = HMAX;
  else h = delta*h;
  for (int i = 0; i < 5; i++) y[i] = y4[i];
  if (!isfinite(y[0]) || !isfinite(y[2]) || !isfinite(y[4])) {{ st = 3; break; }}
  if (y[0] > 30) {{ st = 0; break; }}
}}
out_r[idx] = y[0]; out_th[idx] = y[2]; out_phi[idx] = y[4]; out_st[idx] = st;
"""

# Debug kernel: evaluates the RHS at given states (used to validate the Christoffel port).
_GEO = """
uint idx = thread_position_in_grid.x;
if (idx >= (uint)zin_shape[0]/5) return;
P p;
p.a = prm[0]; p.M = prm[1]; p.r0 = prm[2]; p.th0 = prm[3];
float y0[5];
init_ray(prm[4], prm[5], p, y0);
float z[5], o[5];
for (int i = 0; i < 5; i++) z[i] = zin[idx*5 + i];
geo(z, p, o);
for (int i = 0; i < 5; i++) out[idx*5 + i] = o[i];
"""


def f_paral_gpu(r0, theta0, a, M, alfa, beta):
    """GPU twin of `kerr_core.f_paral`: (r, theta, phi, status), each (len(alfa), len(beta))."""
    alfa = np.asarray(alfa)
    beta = np.asarray(beta)
    na, nb = len(alfa), len(beta)
    res = gb.launch(
        "kerr_trace",
        _HEADER,
        _TRACE,
        {"alfa": alfa, "beta": beta, "prm": np.array([a, M, r0, theta0])},
        {"out_r": np.float32, "out_th": np.float32, "out_phi": np.float32, "out_st": np.int32},
        na * nb,
    )
    shape = (na, nb)
    return (
        res["out_r"].astype(np.float64).reshape(shape),
        res["out_th"].astype(np.float64).reshape(shape),
        res["out_phi"].astype(np.float64).reshape(shape),
        res["out_st"].astype(np.int64).reshape(shape),
    )


def geo_gpu(states, r0, theta0, a, M, alfa, beta):
    """Evaluate the GPU geodesic RHS at `states` (k, 5); for validation against `kc.geo`."""
    states = np.asarray(states, dtype=np.float32).reshape(-1, 5)
    res = gb.launch(
        "kerr_geo",
        _HEADER,
        _GEO,
        {"zin": states.ravel(), "prm": np.array([a, M, r0, theta0, alfa, beta])},
        {"out": np.float32},
        len(states),
        out_len=states.size,
    )
    return res["out"].astype(np.float64).reshape(-1, 5)
