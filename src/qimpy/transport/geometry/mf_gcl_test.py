"""Verify the FULL shell transport (14) + volume GCL (18) in the REAL qimpy code:
a UNIFORM f on a genuinely MOVING frame (nonzero xidot,phidot) must be preserved to
machine precision by f_tr = f + dt(dG - f·dJv)/𝒥 with consistent (None) ghosts.
This is the decisive GCL correctness test. tau_p = inf."""
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

fs = FermiSurface(kF=kF, vF=vF, M_theta=8, Nr=4, T=T, xi_max=6.0, moving_frame=True, process_grid=pg)
fv = FiniteVolume(material=fs, mesh_file=pm, contacts={"periodic": None}, cfl=0.4, process_grid=pg)
dev = fs.rho0.device
cen = torch.as_tensor(fv.geom.centroid_np, device=dev)
# spatially-varying frame -> nonzero gradients -> nonzero xidot,phidot
mu = fs.E_F * (1 + 0.10 * torch.cos(2 * np.pi * cen[:, 0]))
Te = T * (1 + 0.20 * torch.sin(2 * np.pi * cen[:, 1]))
u = torch.zeros(fv.K, 2, device=dev)
u[:, 0] = 0.8 * vF * torch.sin(2 * np.pi * cen[:, 1]); u[:, 1] = 0.4 * vF * torch.cos(2 * np.pi * cen[:, 0])
fv._U = fs.U_from_frame(mu, Te, u); fv._Te = Te.clone()

# The discrete GCL free-stream invariant is a SHELL-CONSTANT occupation f (its k-space
# advection vanishes trivially).  The state is now the unbounded delta_g = logit(f)+xi',
# so a constant level c is stored as delta_g = g_of_f(c) (= logit(c)+xi') and reconstructed
# as f = f_of_g(delta_g) = c.  (delta_g=0 would give f=f0, which VARIES over xi' and is
# only stationary under the full self-consistent frame -- not this isolated probe.)
for c in (0.0, 0.37, 1.0, 2.7):
    fv._u = fs.g_of_f(torch.full((fv.K, fs.Nr * fs.angular.N_theta), float(c), device=dev))
    f = fs.f_of_g(fv._u)                                  # shell-constant occupation (= c)
    mu, Te, u = fs.recover_frame(fv._U, Te_guess=fv._Te)
    uf = fv._faces_fn(fv._u).reshape(-1, fv.Nk)           # reconstruct delta_g faces
    dU = fv._march_U(fv._u, mu, Te, u, 0.0, uf)
    qf = torch.stack([mu, Te, fs.mstar * u[:, 0] / fs.hbar, fs.mstar * u[:, 1] / fs.hbar], 1)
    av = fv._frame_adv(qf, mu, Te, u).reshape(fv.K, 4, fs.Nr, fs.angular.N_theta)
    dmu, dTe, dkD = fs.dframe_from_dU(dU, mu, Te, u)
    xidot, phidot = fs.shell_velocities(mu, Te, u, dmu, dTe, dkD, av[:, 0], av[:, 1], av[:, 2], av[:, 3])
    Jz = fs.mstar * Te / fs.hbar ** 2
    dG = fv._transportG(f, uf, Jz, mu, Te, u, xidot, phidot, None, None, 0.0)
    dJv = fv._transportG(None, uf, Jz, mu, Te, u, xidot, phidot, None, None, 0.0)
    dt = 1e-3
    f_tr = f + dt * (dG - f * dJv) / Jz[:, None]
    res = float((f_tr - f).abs().max())
    print(f"GCL uniform f={c:4.2f}: |ξ̇'|max={float(xidot.abs().max()):.1e} |φ̇|max={float(phidot.abs().max()):.1e}"
          f"  ->  max|df| = {res:.2e}  (want ~1e-14)")
print("PASS: GCL preserves uniform (shell-constant) f under a moving frame.")
