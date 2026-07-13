"""Diagnose the finite-tau_ee blow-up: run the collision setup at 1x CFL and print, each
step, the frame health (Te, k_bar_min), shape deviation, f range, and substep count -- to
see WHAT diverges (collision overshoot? band-bottom small Te/k_bar? something else)."""
import numpy as np, torch, tempfile, os
from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.transport.material import FermiSurface
from qimpy.transport.geometry._finite_volume import FiniteVolume
from qimpy.transport.geometry._mesh import save_mesh
torch.set_default_dtype(torch.float64)
pg = ProcessGrid(rc.comm, "rk", (1, 1))
kF, vF, T = 7.5e-3, 0.11194, 1.33e-5
tmp = tempfile.mkdtemp(); n, L = 20, 1.0
xs = np.linspace(0, L, n + 1); V = np.array([[x, y] for y in xs for x in xs], float)
idx = lambda i, j: j * (n + 1) + i
Tr = [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)] for j in range(n) for i in range(n)] + \
     [[idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)] for j in range(n) for i in range(n)]
be = [[idx(i, 0), idx(i + 1, 0)] for i in range(n)] + [[idx(i, n), idx(i + 1, n)] for i in range(n)] + \
     [[idx(0, j), idx(0, j + 1)] for j in range(n)] + [[idx(n, j), idx(n, j + 1)] for j in range(n)]
pm = os.path.join(tmp, "per.npz")
save_mesh(pm, V, np.array(Tr), np.array(be), ["periodic"] * len(be), lattice=[[L, 0.0], [0.0, L]])
tau_ee = 50.0 * (L / vF)
fs = FermiSurface(kF=kF, vF=vF, M_theta=8, Nr=4, T=T, xi_max=6.0, tau_ee=tau_ee,
                  moving_frame=True, process_grid=pg)
fv = FiniteVolume(material=fs, mesh_file=pm, contacts={"periodic": None}, cfl=0.4, process_grid=pg)
dev = fs.rho0.device
print(f"max|rate_modal|={float(fs.rates_modal.abs().max()):.4e}  1/tau_ee={1/tau_ee:.4e}  "
      f"skip_collision={fv._skip_collision}")
cen = torch.as_tensor(fv.geom.centroid_np, device=dev)
mu = torch.full((fv.K,), fs.E_F, device=dev); Te = T * (1 + 0.05 * torch.cos(2 * np.pi * cen[:, 0]))
u = torch.zeros(fv.K, 2, device=dev); u[:, 0] = 0.3 * vF * torch.sin(2 * np.pi * cen[:, 1])
fv._U = fs.U_from_frame(mu, Te, u); fv._Te = Te.clone(); fv._u = torch.zeros(fv.K, fv.Nk, device=dev)
dt_real = 0.4 * float(fv.geom.inradius.min()) / (fs.v_speed.max().item() + 0.3 * vF)
print(f"dt_real={dt_real:.4e}  max|rate|*dt={float(fs.rates_modal.abs().max())*dt_real:.3e}")
# count substeps by wrapping the CFL helper
orig = fv._cfl_substep; fv._nsub = 0
def counted(*a, **k):
    fv._nsub += 1; return orig(*a, **k)
fv._cfl_substep = counted
for st in range(40):
    fv._nsub = 0
    try:
        fv.step_moving_frame(0.0, dt_real)
    except Exception as e:
        print(f"st{st:3d}: RAISED {type(e).__name__}: {e}"); break
    mu2, Te2, u2 = fs.recover_frame(fv._U, Te_guess=fv._Te)
    f = fs.f_of_g(fv._u); kb = fs._kbar(mu2, Te2)
    fin = torch.isfinite(fv._u).all().item() and torch.isfinite(Te2).all().item()
    cr = float(fv.consistency_residual().max())
    print(f"st{st:3d}: nsub={fv._nsub:3d} consist={cr:.1e} |dg|max={float(fv._u.abs().max()):.2e} "
          f"Te/T=[{float((Te2/T).min()):.2f},{float((Te2/T).max()):.2f}] "
          f"kb_min/kF={float(kb.min())/kF:.3e} |f-f0|={float((f-fs.rho0).abs().max()):.3e} "
          f"f=[{float(f.min()):.1e},{float(f.max()):.4f}] fin={fin}")
    if not fin:
        print("  -> NON-FINITE"); break
print("done")
