"""Fluid-model (macroscopic HLL Riemann) validation.  tau_p=inf.
  (1) PERIODIC: seed a smooth density+shear perturbation, evolve the inviscid Euler
      moments, and confirm N,px,py,E are conserved to machine precision (HLL telescopes).
  (2) DEVICE: source/drain voltage contacts -> a current flows source->drain and the run
      stays finite (inviscid at tau_p=inf accelerates -- no steady state expected).
  (3) TIMING: per-step cost of step_fluid vs the kinetic step_moving_frame on the same mesh.
"""
import numpy as np, torch, tempfile, os, time
from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.transport.material import FermiSurface
from qimpy.transport.geometry._finite_volume import FiniteVolume
from qimpy.transport.geometry._mesh import save_mesh

torch.set_default_dtype(torch.float64)
pg = ProcessGrid(rc.comm, "rk", (1, 1))
kF, vF, T = 7.5e-3, 0.11194, 1.33e-5
tmp = tempfile.mkdtemp()
n, L = 24, 1.0
xs = np.linspace(0, L, n + 1)
V = np.array([[x, y] for y in xs for x in xs], float)
idx = lambda i, j: j * (n + 1) + i
Tr = [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)] for j in range(n) for i in range(n)] + \
     [[idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)] for j in range(n) for i in range(n)]

# ---------- (1) PERIODIC conservation ----------
be = [[idx(i, 0), idx(i + 1, 0)] for i in range(n)] + [[idx(i, n), idx(i + 1, n)] for i in range(n)] + \
     [[idx(0, j), idx(0, j + 1)] for j in range(n)] + [[idx(n, j), idx(n, j + 1)] for j in range(n)]
pm = os.path.join(tmp, "per.npz")
save_mesh(pm, V, np.array(Tr), np.array(be), ["periodic"] * len(be), lattice=[[L, 0.0], [0.0, L]])
fs = FermiSurface(kF=kF, vF=vF, M_theta=8, Nr=4, T=T, xi_max=6.0, fluid_model=True, process_grid=pg)
fv = FiniteVolume(material=fs, mesh_file=pm, contacts={"periodic": None}, cfl=0.4, process_grid=pg)
dev = fs.rho0.device
cen = torch.as_tensor(fv.geom.centroid_np, device=dev)
mu = fs.E_F * (1 + 0.05 * torch.cos(2 * np.pi * cen[:, 0]))          # density wave
Te = torch.full((fv.K,), T, device=dev)
u = torch.zeros(fv.K, 2, device=dev); u[:, 0] = 0.02 * vF * torch.sin(2 * np.pi * cen[:, 1])  # shear
fv._U = fs.U_from_frame(mu, Te, u); fv._Te = Te.clone()
area = torch.as_tensor(fv.geom.area, device=dev)[:, None]
U0 = (area * fv._U).sum(0)
dt = 0.4 * float(fv.geom.inradius.min()) / float(fs.sound_speed(mu, Te).max())
for st in range(80):
    fv.step_fluid(0.0, dt)
U1 = (area * fv._U).sum(0)
Jsc = float((area * fv._U[:, 1:3].abs()).sum()) + 1e-300
dN = abs(U1[0] - U0[0]) / abs(U0[0]); dE = abs(U1[3] - U0[3]) / abs(U0[3])
dpx = abs(U1[1] - U0[1]) / Jsc; dpy = abs(U1[2] - U0[2]) / Jsc
finP = torch.isfinite(fv._U).all().item()
print(f"PERIODIC fluid: cons(N,px,py,E)=({dN:.1e},{dpx:.1e},{dpy:.1e},{dE:.1e}) finite={finP}")

# ---------- (2) DEVICE: source/drain voltage ----------
be2, bm2 = [], []
for i in range(n):
    be2 += [[idx(i, 0), idx(i + 1, 0)]]; bm2 += ["wall"]
    be2 += [[idx(i, n), idx(i + 1, n)]]; bm2 += ["wall"]
for j in range(n):
    be2 += [[idx(0, j), idx(0, j + 1)]]; bm2 += ["source"]
    be2 += [[idx(n, j), idx(n, j + 1)]]; bm2 += ["drain"]
dm = os.path.join(tmp, "dev.npz")
save_mesh(dm, V, np.array(Tr), np.array(be2), bm2)
fs2 = FermiSurface(kF=kF, vF=vF, M_theta=8, Nr=4, T=T, xi_max=6.0, fluid_model=True, process_grid=pg)
fv2 = FiniteVolume(material=fs2, mesh_file=dm,
                   contacts={"source": {"dmu": 0.5 * T}, "drain": {"dmu": -0.5 * T}},
                   cfl=0.4, process_grid=pg)
mu0, Te0, _ = fs2.recover_frame(fv2._U)
dt2 = 0.4 * float(fv2.geom.inradius.min()) / float(fs2.sound_speed(mu0, Te0).max())
for st in range(200):
    fv2.step_fluid(0.0, dt2)
    if st % 50 == 0 or st == 199:
        mu2, Te2, u2 = fs2.recover_frame(fv2._U)
        fin = torch.isfinite(fv2._U).all().item()
        print(f"DEVICE fluid st{st:3d}: <ux>/vF={float(u2[:,0].mean())/vF:+.3e} "
              f"mu/E_F=[{float((mu2/fs2.E_F).min()):.3f},{float((mu2/fs2.E_F).max()):.3f}] finite={fin}")
finD = torch.isfinite(fv2._U).all().item()
ux_pos = float(u2[:, 0].mean()) > 0.0     # source(+dmu, x=0) -> drain: flow in +x

# ---------- (3) TIMING vs kinetic ----------
fsk = FermiSurface(kF=kF, vF=vF, M_theta=8, Nr=4, T=T, xi_max=6.0, moving_frame=True, process_grid=pg)
fvk = FiniteVolume(material=fsk, mesh_file=dm,
                   contacts={"source": {"dmu": 0.5 * T}, "drain": {"dmu": -0.5 * T}},
                   cfl=0.4, process_grid=pg)
muk, Tek, _ = fsk.recover_frame(fvk._U)
dtk = 0.4 * float(fvk.geom.inradius.min()) / fsk.v_speed.max().item()
for _ in range(2):  # warm up
    fv2.step_fluid(0.0, dt2); fvk.step_moving_frame(0.0, dtk)
if dev.type == "cuda": torch.cuda.synchronize()
t0 = time.time()
for _ in range(20): fv2.step_fluid(0.0, dt2)
if dev.type == "cuda": torch.cuda.synchronize()
t_fluid = (time.time() - t0) / 20
t0 = time.time()
for _ in range(20): fvk.step_moving_frame(0.0, dtk)
if dev.type == "cuda": torch.cuda.synchronize()
t_kin = (time.time() - t0) / 20
print(f"TIMING: fluid {t_fluid*1e3:.2f} ms/step  kinetic {t_kin*1e3:.2f} ms/step  "
      f"speedup {t_kin/max(t_fluid,1e-9):.1f}x")

# ---------- (4) SHEAR CONTACT: HLLC should barely diffuse a stationary u_x(y) layer ----------
# Uniform density/pressure, u_y=0, x-velocity shear varying in y: a contact discontinuity
# (tangential-velocity jump) that is a stationary Euler solution.  HLLC rides it on the
# contact wave S_* (near-zero diffusion); HLL would smear it with acoustic dissipation.
fs3 = FermiSurface(kF=kF, vF=vF, M_theta=8, Nr=4, T=T, xi_max=6.0, fluid_model=True, process_grid=pg)
fv3 = FiniteVolume(material=fs3, mesh_file=pm, contacts={"periodic": None}, cfl=0.4, process_grid=pg)
mu3 = torch.full((fv3.K,), fs3.E_F, device=dev)                      # uniform p,rho
Te3 = torch.full((fv3.K,), T, device=dev)
u3 = torch.zeros(fv3.K, 2, device=dev)
u3[:, 0] = 0.05 * vF * torch.sin(2 * np.pi * cen[:, 1])   # smooth (resolved) shear layer
fv3._U = fs3.U_from_frame(mu3, Te3, u3); fv3._Te = Te3.clone()
ux0 = u3[:, 0].clone()
dt3 = 0.4 * float(fv3.geom.inradius.min()) / float(fs3.sound_speed(mu3, Te3).max())
for st in range(50):
    fv3.step_fluid(0.0, dt3)
_, _, u3b = fs3.recover_frame(fv3._U)
shear_drift = float((u3b[:, 0] - ux0).abs().max()) / (0.05 * vF)
print(f"SHEAR contact (HLLC): stationary u_x(y) layer relative drift after 50 steps = {shear_drift:.2e}")

ok = finP and finD and ux_pos and dN < 1e-12 and dpx < 1e-12 and dE < 1e-12
print("PASS: fluid model (HLLC) conserves (periodic) + stable device current + preserves shear contact"
      if ok else "CHECK")
