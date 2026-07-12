"""Isolate the drift-0.3 collisional NaN: run BALLISTIC (tau_ee=inf) drift-0.3 shear
with per-step Te / mu-Te / |f-f0| / finiteness, to see whether the collision matters
and what actually diverges."""
import numpy as np, torch, tempfile, os
from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.transport.material import FermiSurface
from qimpy.transport.geometry._finite_volume import FiniteVolume
from qimpy.transport.geometry._mesh import save_mesh

torch.set_default_dtype(torch.float64)
pg = ProcessGrid(rc.comm, "rk", (1, 1))
kF, vF, T = 7.5e-3, 0.11194, 1.33e-5
tmp = tempfile.mkdtemp()
n, L = 20, 1.0
xs = np.linspace(0, L, n + 1)
V = np.array([[x, y] for y in xs for x in xs], float)
idx = lambda i, j: j * (n + 1) + i
Tr = [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)] for j in range(n) for i in range(n)] + \
     [[idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)] for j in range(n) for i in range(n)]
be = [[idx(i, 0), idx(i + 1, 0)] for i in range(n)] + [[idx(i, n), idx(i + 1, n)] for i in range(n)] + \
     [[idx(0, j), idx(0, j + 1)] for j in range(n)] + [[idx(n, j), idx(n, j + 1)] for j in range(n)]
pm = os.path.join(tmp, "per.npz")
save_mesh(pm, V, np.array(Tr), np.array(be), ["periodic"] * len(be), lattice=[[L, 0.0], [0.0, L]])


def krate(fv, fs):
    mu, Te, u = fs.recover_frame(fv._U, Te_guess=fv._Te)
    uf = fv._faces_fn(fv._u).reshape(-1, fv.Nk)
    dU = fv._march_U(fv._u, mu, Te, u, 0.0, uf)
    qf = torch.stack([mu, Te, fs.mstar * u[:, 0] / fs.hbar, fs.mstar * u[:, 1] / fs.hbar], 1)
    av = fv._frame_adv(qf, mu, Te, u).reshape(fv.K, 4, fs.Nr, fs.angular.N_theta)
    dmu, dTe, dkD = fs.dframe_from_dU(dU, mu, Te, u)
    xid, phid = fs.shell_velocities(mu, Te, u, dmu, dTe, dkD, av[:, 0], av[:, 1], av[:, 2], av[:, 3])
    dxi = float(torch.diff(fs.radial.xi).abs().min())
    return float(xid.abs().max()) / dxi + float(phid.abs().max()) / fs.angular.wphi


for tau_ee, tag in [(np.inf, "BALLISTIC tau_ee=inf"), (50.0 * (L / vF), "tau_ee=50 L/vF")]:
    fs = FermiSurface(kF=kF, vF=vF, M_theta=8, Nr=4, T=T, xi_max=6.0, tau_ee=tau_ee,
                      moving_frame=True, process_grid=pg)
    fv = FiniteVolume(material=fs, mesh_file=pm, contacts={"periodic": None}, cfl=0.4, process_grid=pg)
    dev = fs.rho0.device
    cen = torch.as_tensor(fv.geom.centroid_np, device=dev)
    mu = torch.full((fv.K,), fs.E_F, device=dev); Te = T * (1 + 0.05 * torch.cos(2 * np.pi * cen[:, 0]))
    u = torch.zeros(fv.K, 2, device=dev); u[:, 0] = 0.3 * vF * torch.sin(2 * np.pi * cen[:, 1])
    fv._U = fs.U_from_frame(mu, Te, u); fv._Te = Te.clone()
    fv._u = torch.zeros((fv.K, fs.Nr * fs.angular.N_theta), device=dev)   # delta_g=0 <=> f=f0
    dt_real = 0.4 * float(fv.geom.inradius.min()) / (fs.v_speed.max().item() + 0.3 * vF)
    print(f"\n=== {tag}  skip_collision={fv._skip_collision} ===")
    for st in range(121):
        if st % 15 == 0:
            mu, Te, u = fs.recover_frame(fv._U, Te_guess=fv._Te)
            ff = fs.f_of_g(fv._u)                          # occupation from unbounded delta_g
            fin = torch.isfinite(fv._u).all().item() and torch.isfinite(Te).all().item()
            print(f" st{st:3d}: Te/T=[{(Te/T).min():.2f},{(Te/T).max():.2f}] mu/Te_min={(mu/Te).min():.2f}"
                  f" |f-f0|={float((ff-fs.rho0).abs().max()):.2e} f=[{float(ff.min()):.1e},{float(ff.max()):.6f}]"
                  f" fin={fin}")
            if not fin:
                break
        fv.step_moving_frame(0.0, min(dt_real, 0.3 / max(krate(fv, fs), 1e-30)))
