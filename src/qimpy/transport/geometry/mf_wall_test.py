"""Diffuse-wall mass conservation.  An ALL-WALL box (no contacts) must conserve total N
EXACTLY for ANY specularity -- nothing can leave.  Seed a drifted+heated state and run;
report the total-mass drift for specularity = 1.0 (specular) and 0.5 (diffuse).  tau_p=inf."""
import numpy as np, torch, tempfile, os
from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.transport.material import FermiSurface
from qimpy.transport.geometry._finite_volume import FiniteVolume
from qimpy.transport.geometry._mesh import save_mesh
torch.set_default_dtype(torch.float64)
pg = ProcessGrid(rc.comm, "rk", (1, 1))
kF, vF, T = 7.5e-3, 0.11194, 1.33e-5
tmp = tempfile.mkdtemp(); n, L = 16, 1.0
xs = np.linspace(0, L, n + 1); V = np.array([[x, y] for y in xs for x in xs], float)
idx = lambda i, j: j * (n + 1) + i
Tr = [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)] for j in range(n) for i in range(n)] + \
     [[idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)] for j in range(n) for i in range(n)]
be = [[idx(i, 0), idx(i + 1, 0)] for i in range(n)] + [[idx(i, n), idx(i + 1, n)] for i in range(n)] + \
     [[idx(0, j), idx(0, j + 1)] for j in range(n)] + [[idx(n, j), idx(n, j + 1)] for j in range(n)]
wm = os.path.join(tmp, "box.npz")
save_mesh(wm, V, np.array(Tr), np.array(be), ["wall"] * len(be))   # ALL walls

for s in (1.0, 0.5):
    fs = FermiSurface(kF=kF, vF=vF, M_theta=8, Nr=4, T=T, xi_max=6.0, specularity=s,
                      moving_frame=True, process_grid=pg)
    fv = FiniteVolume(material=fs, mesh_file=wm, contacts={"wall": None}, cfl=0.4, process_grid=pg)
    dev = fs.rho0.device
    cen = torch.as_tensor(fv.geom.centroid_np, device=dev)
    mu = fs.E_F * (1 + 0.03 * torch.cos(2 * np.pi * cen[:, 0]))
    Te = T * (1 + 0.05 * torch.cos(2 * np.pi * cen[:, 1]))
    u = torch.zeros(fv.K, 2, device=dev)
    u[:, 0] = 0.05 * vF * torch.sin(2 * np.pi * cen[:, 1])          # drift with wall-normal comp.
    u[:, 1] = 0.05 * vF * torch.cos(2 * np.pi * cen[:, 0])
    fv._U = fs.U_from_frame(mu, Te, u); fv._Te = Te.clone()
    fv._u = torch.zeros(fv.K, fv.Nk, device=dev)
    area = torch.as_tensor(fv.geom.area, device=dev)
    N0 = float((area * fv._U[:, 0]).sum())
    dt = 0.4 * float(fv.geom.inradius.min()) / (fs.v_speed.max().item() + 0.05 * vF)
    for st in range(60):
        fv.step_moving_frame(0.0, dt)
    N1 = float((area * fv._U[:, 0]).sum())
    fin = torch.isfinite(fv._U).all().item()
    print(f"specularity={s:.2f}: |dN|/N = {abs(N1 - N0) / abs(N0):.3e}  finite={fin}")
print("(both should be ~1e-15 if walls conserve mass exactly)")
