"""Inviscid fluid-model (HLLC) run in the square source-drain channel; checkpoint + plot.
Source (left) dmu=+0.5T, drain (right) dmu=-0.5T, specular walls top/bottom.  tau_p=inf."""
import numpy as np, torch, tempfile, os, shutil
from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.io import Checkpoint, CheckpointPath, CheckpointContext
from qimpy.transport.material import FermiSurface
from qimpy.transport.geometry._finite_volume import FiniteVolume
from qimpy.transport.geometry._mesh import save_mesh
from qimpy.transport.plot import run_finite_volume
torch.set_default_dtype(torch.float64)
pg = ProcessGrid(rc.comm, "rk", (1, 1))
kF, vF, T = 7.5e-3, 0.11194, 1.33e-5
tmp = tempfile.mkdtemp(); n, L = 24, 1.0
xs = np.linspace(0, L, n + 1)
V = np.array([[x, y] for y in xs for x in xs], float)
idx = lambda i, j: j * (n + 1) + i
Tr = [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)] for j in range(n) for i in range(n)] + \
     [[idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)] for j in range(n) for i in range(n)]
be, bm = [], []
for i in range(n):
    be += [[idx(i, 0), idx(i + 1, 0)]]; bm += ["wall"]
    be += [[idx(i, n), idx(i + 1, n)]]; bm += ["wall"]
for j in range(n):
    be += [[idx(0, j), idx(0, j + 1)]]; bm += ["source"]
    be += [[idx(n, j), idx(n, j + 1)]]; bm += ["drain"]
dm = os.path.join(tmp, "sq.npz")
save_mesh(dm, V, np.array(Tr), np.array(be), bm)

fs = FermiSurface(kF=kF, vF=vF, M_theta=8, Nr=4, T=T, xi_max=6.0, fluid_model=True, process_grid=pg)
fv = FiniteVolume(material=fs, mesh_file=dm,
                  contacts={"source": {"dmu": 0.5 * T}, "drain": {"dmu": -0.5 * T}},
                  cfl=0.4, process_grid=pg)
mu0, Te0, _ = fs.recover_frame(fv._U)
dt = 0.4 * float(fv.geom.inradius.min()) / float(fs.sound_speed(mu0, Te0).max())
nsteps = 150
for st in range(nsteps + 1):
    if st % 30 == 0:
        fv.update_stash(st, st * dt)
    if st < nsteps:
        fv.step_fluid(0.0, dt)
mu, Te, u = fs.recover_frame(fv._U)
print(f"final step {nsteps}: <ux>/vF={float(u[:,0].mean())/vF:+.3e} <uy>/vF={float(u[:,1].mean())/vF:+.1e} "
      f"mu/E_F=[{float((mu/fs.E_F).min()):.3f},{float((mu/fs.E_F).max()):.3f}] finite={torch.isfinite(fv._U).all().item()}")

ckpt = os.path.join(tmp, "out.h5")
with Checkpoint(ckpt, writable=True, rotate=False) as cp:
    fv._save_checkpoint(CheckpointPath(cp, "geometry"), CheckpointContext(""))
out = os.path.join(tmp, "sqfluid_{}.png")
run_finite_volume([ckpt], slice(None), out, {"field": "n"},
                  {"density": 1.5, "linewidth": 1.0, "arrowsize": 1.0}, 130)
pngs = sorted([p for p in os.listdir(tmp) if p.endswith(".png")])
print("PNGs:", pngs)
shutil.copy(os.path.join(tmp, pngs[-1]), "/tmp/sqfluid_final.png")
print("saved /tmp/sqfluid_final.png")
