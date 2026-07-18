"""Does the INVISCID FLUID model (step_fluid, HLLC, no shell f) handle walls / diffuse walls?
All-wall box, drifted+heated seed, tau_p=inf.  Check (a) mass conservation, (b) whether the
specularity parameter changes anything (it should NOT in the inviscid model -- a diffuse wall's
drag/heat are viscous effects absent from Euler; only the specular slip wall is consistent)."""
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
save_mesh(wm, V, np.array(Tr), np.array(be), ["wall"] * len(be))

Ufinal = {}
for s in (1.0, 0.5):
    fs = FermiSurface(kF=kF, vF=vF, M_theta=8, Nr=4, T=T, xi_max=6.0, specularity=s,
                      fluid_model=True, process_grid=pg)
    fv = FiniteVolume(material=fs, mesh_file=wm, contacts={"wall": None}, cfl=0.4, process_grid=pg)
    dev = fs.rho0.device
    cen = torch.as_tensor(fv.geom.centroid_np, device=dev)
    mu = fs.E_F * (1 + 0.03 * torch.cos(2 * np.pi * cen[:, 0]))
    Te = T * (1 + 0.05 * torch.cos(2 * np.pi * cen[:, 1]))
    u = torch.zeros(fv.K, 2, device=dev)
    u[:, 0] = 0.05 * vF * torch.sin(2 * np.pi * cen[:, 1])
    u[:, 1] = 0.05 * vF * torch.cos(2 * np.pi * cen[:, 0])
    fv._U = fs.U_from_frame(mu, Te, u); fv._Te = Te.clone()
    area = torch.as_tensor(fv.geom.area, device=dev)
    N0 = float((area * fv._U[:, 0]).sum()); E0 = float((area * fv._U[:, 3]).sum())
    mu0, Te0, _ = fs.recover_frame(fv._U)
    dt = 0.4 * float(fv.geom.inradius.min()) / float(fs.sound_speed(mu0, Te0).max())
    for st in range(80):
        fv.step_fluid(0.0, dt)
    N1 = float((area * fv._U[:, 0]).sum()); E1 = float((area * fv._U[:, 3]).sum())
    fin = torch.isfinite(fv._U).all().item()
    Ufinal[s] = fv._U.clone()
    print(f"specularity={s:.2f}: |dN|/N={abs(N1-N0)/abs(N0):.3e}  |dE|/E={abs(E1-E0)/abs(E0):.3e}  finite={fin}")

diff = float((Ufinal[1.0] - Ufinal[0.5]).abs().max()) / float(Ufinal[1.0].abs().max())
print(f"\nmax|U(s=1.0) - U(s=0.5)| / |U| = {diff:.3e}")
print("=> specularity is IGNORED by the inviscid fluid model" if diff < 1e-14
      else "=> specularity DOES change the fluid result")
