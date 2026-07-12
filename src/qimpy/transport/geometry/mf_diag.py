"""Diagnose the strong-drift consistency degradation: track Te heating, mu/Te
(band-bottom margin), per-channel consistency, and f overshoot over the run."""
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
n, L = 24, 1.0
xs = np.linspace(0, L, n + 1)
V = np.array([[x, y] for y in xs for x in xs], float)
idx = lambda i, j: j * (n + 1) + i
Tr = [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)] for j in range(n) for i in range(n)] + \
     [[idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)] for j in range(n) for i in range(n)]
be = [[idx(i, 0), idx(i + 1, 0)] for i in range(n)] + [[idx(i, n), idx(i + 1, n)] for i in range(n)] + \
     [[idx(0, j), idx(0, j + 1)] for j in range(n)] + [[idx(n, j), idx(n, j + 1)] for j in range(n)]
pm = os.path.join(tmp, "per.npz")
save_mesh(pm, V, np.array(Tr), np.array(be), ["periodic"] * len(be), lattice=[[L, 0.0], [0.0, L]])

for drift, iters in [(3.0, 6), (3.0, 20)]:
    fs = FermiSurface(kF=kF, vF=vF, M_theta=8, Nr=4, T=T, xi_max=6.0, moving_frame=True, process_grid=pg)
    fv = FiniteVolume(material=fs, mesh_file=pm, contacts={"periodic": None}, cfl=0.4, process_grid=pg)
    dev = fs.rho0.device
    cen = torch.as_tensor(fv.geom.centroid_np, device=dev)
    mu = torch.full((fv.K,), fs.E_F, device=dev); Te = T * (1 + 0.05 * torch.cos(2 * np.pi * cen[:, 0]))
    u = torch.zeros(fv.K, 2, device=dev); u[:, 0] = drift * vF * torch.sin(2 * np.pi * cen[:, 1])
    fv._U = fs.U_from_frame(mu, Te, u); fv._Te = Te.clone(); fv._u = fs.rho0[None, :].repeat(fv.K, 1).clone()
    dt = 0.4 * float(fv.geom.inradius.min()) / (fs.v_speed.max().item() + drift * vF)
    print(f"\n=== drift={drift} recover_iters={iters} dt={dt:.2e} ===")
    for st in range(201):
        if st % 40 == 0:
            mu, Te, u = fs.recover_frame(fv._U, Te_guess=fv._Te, iters=iters)
            Uf = fs.moments_of_f(fv._u, mu, Te, u)
            sc = fv._U.abs().mean(0).clamp_min(1e-30)
            cr = ((Uf - fv._U).abs() / sc).max(0).values
            print(f" st{st:3d}: Te/T=[{(Te/T).min():.2f},{(Te/T).max():.2f}] mu/Te_min={(mu/Te).min():.1f}"
                  f" consist[N,px,py,E]=[{cr[0]:.1e},{cr[1]:.1e},{cr[2]:.1e},{cr[3]:.1e}]"
                  f" f=[{fv._u.min():.1e},{fv._u.max():.4f}]")
        if st < 200:
            fv.step_moving_frame(0.0, dt)
