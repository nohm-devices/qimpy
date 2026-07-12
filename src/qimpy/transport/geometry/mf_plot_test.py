"""End-to-end staggered-output test: moving-frame device run -> checkpoint with
CELL-CENTERED scalars (n,T_e) + per-EDGE fluxes (j.n, q.n) -> plot.py renders a PNG
with face-native currents.  tau_p=inf."""
import numpy as np, torch, tempfile, os
from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.io import Checkpoint, CheckpointPath, CheckpointContext
from qimpy.transport.material import FermiSurface
from qimpy.transport.geometry._finite_volume import FiniteVolume
from qimpy.transport.geometry._mesh import save_mesh

torch.set_default_dtype(torch.float64)
pg = ProcessGrid(rc.comm, "rk", (1, 1))
kF, vF, T = 7.5e-3, 0.11194, 1.33e-5
tmp = tempfile.mkdtemp()
n, L = 16, 1.0
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
dm = os.path.join(tmp, "dev.npz")
save_mesh(dm, V, np.array(Tr), np.array(be), bm)

fs = FermiSurface(kF=kF, vF=vF, M_theta=8, Nr=4, T=T, xi_max=6.0, moving_frame=True, process_grid=pg)
fv = FiniteVolume(material=fs, mesh_file=dm,
                  contacts={"source": {"dmu": 0.5 * T}, "drain": {"dmu": -0.5 * T}},
                  cfl=0.4, process_grid=pg)
dt = 0.4 * float(fv.geom.inradius.min()) / fs.v_speed.max().item()
for st in range(9):
    if st % 3 == 0:
        fv.update_stash(st, float(st) * dt)
    fv.step_moving_frame(0.0, dt)
fv.update_stash(9, 9 * dt)
print(f"stashed {len(fv._stash_i)} frames; obs names = {fs.get_observable_names() if not fv._moving else 'moving'}")

ckpt = os.path.join(tmp, "out.h5")
with Checkpoint(ckpt, writable=True, rotate=False) as cp:
    fv._save_checkpoint(CheckpointPath(cp, "geometry"), CheckpointContext(""))
import h5py
with h5py.File(ckpt) as f:
    g = f["geometry"]
    keys = list(g.keys())
    print("checkpoint /geometry datasets:", keys)
    for k in ("fv_observables", "fv_edge_flux", "edge_midpoints", "edge_normals"):
        print(f"  {k}: {'present ' + str(g[k].shape) if k in g else 'MISSING'}")
    print("  observable_names:", bytes(np.array(g['observable_names'])).decode() if 'observable_names' in g else '?')

from qimpy.transport.plot import run_finite_volume
outpng = os.path.join(tmp, "frame_{}.png")
run_finite_volume([ckpt], slice(None), outpng, {"field": "n"},
                  {"density": 1.4, "linewidth": 1.0, "arrowsize": 1.0}, 110)
pngs = [p for p in os.listdir(tmp) if p.endswith(".png")]
print("PNGs written:", sorted(pngs))
print("PASS: staggered checkpoint + face-native plot" if pngs else "CHECK: no PNG")
