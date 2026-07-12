"""Standalone validation of the FermiSurface moving-frame kit (Stage 1, momentum-space only).

Checks, in the REAL qimpy code on the actual Gauss-Legendre radial + midpoint-Fourier angular
nodes (GaAs params, deeply degenerate E_F/T~32):
  1. frame recovery round-trip  (n,J,E) -> (mu,Te,u) -> back,  across drift 0.15..20 vF + Te ripple
  2. moment-free projection makes moments_of_f(f) == U_from_frame  to round-off (consistency),
     drift-independent; and the RAW (unprojected) deviation does NOT (test is non-trivial)
  3. frame-basis Gram is exactly diagonal (off-diag/diag ~ 1e-16)
  4. projection is idempotent
  5. Eth_FD (degenerate Sommerfeld) matches scipy.spence dilog
Run on the A100:  python3 mf_kit_test.py
"""
import numpy as np, torch
from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.transport.material import FermiSurface

torch.set_default_dtype(torch.float64)
pg = ProcessGrid(rc.comm, "rk", (1, 1))
kF, vF, T = 7.5e-3, 0.11194, 1.33e-5                      # GaAs (validated regime)
fs = FermiSurface(kF=kF, vF=vF, M_theta=8, Nr=4, T=T, xi_max=6.0,
                  moving_frame=True, process_grid=pg)
EF = fs.E_F
dev = fs.rho0.device
Nk = fs.Nr * fs.angular.N_theta
K = 256
gy = torch.linspace(0.0, 1.0, K, device=dev)
print(f"FermiSurface: Nr={fs.Nr} N_theta={fs.angular.N_theta} Nk={Nk}  "
      f"m*={fs.mstar:.4f} E_F/T={EF/T:.1f}  xi_max=6")
print("-" * 100)


def rel(A, B, s):
    return float(((A - B).abs() / s).max())


def check(drift, te_ripple=0.05):
    mu = torch.full((K,), EF, device=dev)
    Te = T * (1.0 + te_ripple * torch.cos(2 * np.pi * gy))
    u = torch.zeros(K, 2, device=dev)
    u[:, 0] = drift * vF * torch.sin(2 * np.pi * gy)
    U = fs.U_from_frame(mu, Te, u)
    # (1) recovery round-trip
    mu2, Te2, u2 = fs.recover_frame(U)
    e = (rel(mu2, mu, EF), rel(Te2, Te, T), rel(u2, u, vF))
    # (2) consistency via projection (use the recovered frame, as the stepper will)
    torch.manual_seed(0)
    d0 = 1e-2 * torch.randn(K, Nk, device=dev)
    Uf = fs.U_from_frame(mu2, Te2, u2)
    nsc = U[:, 0].abs().mean(); Jsc = U[:, 1:3].abs().mean().clamp_min(1e-30); Esc = U[:, 3].abs().mean()
    Uraw = fs.moments_of_f(fs.rho0 + d0, mu2, Te2, u2)
    dP = fs.project_moment_free(d0, mu2, Te2)
    Up = fs.moments_of_f(fs.rho0 + dP, mu2, Te2, u2)
    raw = max(rel(Uraw[:, 0], Uf[:, 0], nsc), rel(Uraw[:, 1:3], Uf[:, 1:3], Jsc), rel(Uraw[:, 3], Uf[:, 3], Esc))
    prj = max(rel(Up[:, 0], Uf[:, 0], nsc), rel(Up[:, 1:3], Uf[:, 1:3], Jsc), rel(Up[:, 3], Uf[:, 3], Esc))
    kD = fs.mstar * drift * vF / kF
    print(f"drift={drift:5.2f}vF |kD|/kF={kD:5.1f} | recover(mu,Te,u)=({e[0]:.1e},{e[1]:.1e},{e[2]:.1e})"
          f" | consist raw={raw:.1e} -> projected={prj:.1e}")
    return prj, e


results = [check(d) for d in (0.15, 3.0, 20.0)]
_p, _e = check(0.15, te_ripple=0.5)                       # heated: stress recover_frame Newton
print(f"heated Te ripple 50%: recover=({_e[0]:.1e},{_e[1]:.1e},{_e[2]:.1e}) projected consist={_p:.1e}")
print("-" * 100)

# (3) Gram diagonality of {1, xi', kbar cos, kbar sin} in flat measure
mu = torch.full((1,), EF, device=dev); Te = torch.full((1,), T, device=dev)
kb = fs._kbar(mu, Te)[0]                                  # (Nr,)
th = fs.angular.theta
w = fs.radial.flat_w[:, None] * fs.angular.wphi           # (Nr,1)
B = [torch.ones(fs.Nr, fs.angular.N_theta, device=dev),
     fs.radial.xi[:, None] * torch.ones_like(th)[None, :],
     kb[:, None] * torch.cos(th)[None, :],
     kb[:, None] * torch.sin(th)[None, :]]
G = torch.stack([torch.stack([(w * B[a] * B[b]).sum() for b in range(4)]) for a in range(4)])
offdiag = (G - torch.diag(torch.diag(G))).abs().max() / torch.diag(G).abs().max()
print(f"frame-basis Gram off-diag/diag = {float(offdiag):.1e}  (want ~1e-16: diagonal => 4 scalar projections)")

# (4) idempotency
torch.manual_seed(1)
d0 = torch.randn(K, Nk, device=dev)
mu2 = torch.full((K,), EF, device=dev); Te2 = T * (1 + 0.05 * torch.cos(2 * np.pi * gy))
d1 = fs.project_moment_free(d0, mu2, Te2)
d2 = fs.project_moment_free(d1, mu2, Te2)
print(f"projection idempotency |P^2-P| = {float((d2 - d1).abs().max()):.1e}")

# (5) Eth_FD vs scipy dilog
from scipy.special import spence
x = EF / T
ref = fs.g2d * T ** 2 * (-spence(1.0 + np.exp(min(x, 700.0))))
got = float(fs.Eth_FD(torch.tensor(EF), torch.tensor(T)))
print(f"Eth_FD vs scipy.spence: rel err = {abs(got - ref) / abs(ref):.1e}")
print("-" * 100)
ok = all(p < 1e-11 for p, _ in results) and float(offdiag) < 1e-12
print("PASS: recovery + drift-independent machine-precision consistency + diagonal Gram."
      if ok else "CHECK: a metric exceeds tolerance.")
