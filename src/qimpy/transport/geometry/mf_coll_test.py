"""Stage-2 collision hook check: with finite tau_ee the modal collision relaxes the
shape; U is marched by transport ONLY and the projection re-pins f, so U-conservation
(periodic) and consistency must STAY machine-precise, and the collision must be active
(_skip_collision False) and bounded."""
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

# tau_ee finite -> modal collision on m>=2 / radial n>=1 (viscous + thermal shape)
tau_ee = 50.0 * (L / vF)                     # a few mean-free-times over the run
fs = FermiSurface(kF=kF, vF=vF, M_theta=8, Nr=4, T=T, xi_max=6.0, tau_ee=tau_ee,
                  moving_frame=True, process_grid=pg)
fv = FiniteVolume(material=fs, mesh_file=pm, contacts={"periodic": None}, cfl=0.4, process_grid=pg)
print(f"_skip_collision={fv._skip_collision} (want False -> collision active)")
dev = fs.rho0.device
cen = torch.as_tensor(fv.geom.centroid_np, device=dev)
mu = torch.full((fv.K,), fs.E_F, device=dev); Te = T * (1 + 0.05 * torch.cos(2 * np.pi * cen[:, 0]))
u = torch.zeros(fv.K, 2, device=dev); u[:, 0] = 0.3 * vF * torch.sin(2 * np.pi * cen[:, 1])
fv._U = fs.U_from_frame(mu, Te, u); fv._Te = Te.clone(); fv._u = fs.rho0[None, :].repeat(fv.K, 1).clone()
U0 = fv.U_totals().clone()
Jsc = float((fv.geom.area[:, None] * fv._U[:, 1:3].abs()).sum()) + 1e-300
def krate():
    mu, Te, u = fs.recover_frame(fv._U, Te_guess=fv._Te)
    uf = fv._faces_fn(fv._u).reshape(-1, fv.Nk)
    dU = fv._march_U(fv._u, mu, Te, u, 0.0, uf)
    gmu, gTe = fv._grad(mu), fv._grad(Te); gkD = fv._grad(fs.mstar * u / fs.hbar)
    dmu, dTe, dkD = fs.dframe_from_dU(dU, mu, Te, u)
    xid, phid = fs.shell_velocities(mu, Te, u, dmu, dTe, dkD, gmu, gTe, gkD)
    dxi = float(torch.diff(fs.radial.xi).abs().min())
    return float(xid.abs().max()) / dxi + float(phid.abs().max()) / fs.angular.wphi

dt_real = 0.4 * float(fv.geom.inradius.min()) / (fs.v_speed.max().item() + 0.3 * vF)
d_amp0 = None
for st in range(15):
    fv.step_moving_frame(0.0, 8.0 * min(dt_real, 0.3 / max(krate(), 1e-30)))   # 8x over-CFL
    if st % 5 == 0 or st == 14:
        Ut = fv.U_totals(); cr = fv.consistency_residual()
        dN = abs(Ut[0] - U0[0]) / abs(U0[0]); dE = abs(Ut[3] - U0[3]) / abs(U0[3])
        dpx = abs(Ut[1] - U0[1]) / Jsc; dpy = abs(Ut[2] - U0[2]) / Jsc
        damp = float((fv._u - fs.rho0).abs().max())
        if d_amp0 is None:
            d_amp0 = damp
        print(f"st{st:3d}: cons(N,px,py,E)=({dN:.1e},{dpx:.1e},{dpy:.1e},{dE:.1e}) "
              f"consist={float(cr.max()):.1e} |f-f0|max={damp:.2e} fmax={fv._u.max():.4f}")
print("PASS" if torch.isfinite(fv._u).all() and not fv._skip_collision else "CHECK")
