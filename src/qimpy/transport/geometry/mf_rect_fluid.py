"""Inviscid fluid-model (HLLC) on the qimpy rect-domain example geometry, run to STEADY STATE.
Source circle @(10,55) dmu=+0.1, drain circle @(10,5) dmu=-0.1, specular walls, tau_p=inf,
no B-field.  Final plot made with qimpy's own plotter (run_finite_volume)."""
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
tmp = tempfile.mkdtemp()

# --- rect-domain example mesh (examples/Transport/make_rect_mesh.py, h=3.0) ---
X0, X1, Y0, Y1 = 5., 105., 5., 55.; SRC, DRN = (10., 55., 5.), (10., 5., 5.); h = 3.0
ny = max(2, 2 * round(0.5 * (Y1 - Y0) / h)); nx = 2 * ny
xs = np.linspace(X0, X1, nx + 1); ys = np.linspace(Y0, Y1, ny + 1)
Vl = [[x, y] for y in ys for x in xs]; vid = lambda i, j: j * (nx + 1) + i; T = []
for j in range(ny):
    for i in range(nx):
        a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1); cen = len(Vl)
        Vl.append([0.5 * (xs[i] + xs[i + 1]), 0.5 * (ys[j] + ys[j + 1])])
        T += [[a, b, cen], [b, c, cen], [c, d, cen], [d, a, cen]]
Vv = np.array(Vl, float)
edges = ([(vid(i, 0), vid(i + 1, 0)) for i in range(nx)] + [(vid(i, ny), vid(i + 1, ny)) for i in range(nx)] +
         [(vid(0, j), vid(0, j + 1)) for j in range(ny)] + [(vid(nx, j), vid(nx, j + 1)) for j in range(ny)])
be, bm = [], []
for a, b in edges:
    mx, my = 0.5 * (Vv[a] + Vv[b])
    bm.append("source" if (mx - SRC[0]) ** 2 + (my - SRC[1]) ** 2 <= SRC[2] ** 2 else
              "drain" if (mx - DRN[0]) ** 2 + (my - DRN[1]) ** 2 <= DRN[2] ** 2 else "wall"); be.append([a, b])
dm = os.path.join(tmp, "rect.npz"); save_mesh(dm, Vv, np.array(T, int), np.array(be, int), bm)
print(f"mesh: {len(T)} tris, src={bm.count('source')} drn={bm.count('drain')} wall={bm.count('wall')}")

# --- inviscid fluid on the rect geometry (degenerate, example length/velocity scale) ---
kF, vF, Tt = 1.0, 1.5, 0.05
fs = FermiSurface(kF=kF, vF=vF, M_theta=12, Nr=1, T=Tt, xi_max=6.0, fluid_model=True, process_grid=pg)
fv = FiniteVolume(material=fs, mesh_file=dm,
                  contacts={"source": {"dmu": 0.1}, "drain": {"dmu": -0.1}}, cfl=0.4, process_grid=pg)
mu0, Te0, _ = fs.recover_frame(fv._U)
dt = 0.4 * float(fv.geom.inradius.min()) / float(fs.sound_speed(mu0, Te0).max())
print(f"E_F={fs.E_F:.4f} c_s={float(fs.sound_speed(mu0,Te0).max()):.3f} dt={dt:.4f}")

# density-gradient probes: source region @(10,55), drain region @(10,5)
cen = torch.as_tensor(fv.geom.centroid_np, device=fv._U.device)
src_m = ((cen[:, 0] - 10.) ** 2 + (cen[:, 1] - 55.) ** 2) < 15. ** 2
drn_m = ((cen[:, 0] - 10.) ** 2 + (cen[:, 1] - 5.) ** 2) < 15. ** 2

# --- run to steady state (stop when the per-block relative change of U plateaus below tol) ---
prev = fv._U.clone(); tsim = 0.0; steps = 0
for block in range(120):
    for _ in range(50):
        fv.step_fluid(tsim, dt); tsim += dt; steps += 1
    ch = float((fv._U - prev).abs().max()) / (float(fv._U.abs().max()) + 1e-300)
    mu, Te, u = fs.recover_frame(fv._U); n = fv._U[:, 0]
    ngrad = (float(n[src_m].mean()) - float(n[drn_m].mean())) / float(n.mean())
    print(f"block{block:3d} steps={steps:5d} rel_change={ch:.2e} <|u|>/vF={float(u.norm(dim=1).mean())/vF:.3e} "
          f"umax/vF={float(u.norm(dim=1).max())/vF:.3e} (n_src-n_drn)/n={ngrad:+.3e} fin={torch.isfinite(fv._U).all().item()}")
    if ch < 1e-7:
        print(f"*** STEADY STATE at step {steps} (rel_change {ch:.1e}) ***"); break
    prev = fv._U.clone()
n = fv._U[:, 0]; imn = int(n.argmin()); imx = int(n.argmax())
print(f"FINAL density: mean={float(n.mean()):.4e}  (n_src-n_drn)/n={(float(n[src_m].mean())-float(n[drn_m].mean()))/float(n.mean()):+.3e}")
print(f"  n_MIN at (x={float(cen[imn,0]):.0f},y={float(cen[imn,1]):.0f}) = {float(n[imn]):.4e}   "
      f"n_MAX at (x={float(cen[imx,0]):.0f},y={float(cen[imx,1]):.0f}) = {float(n[imx]):.4e}   "
      f"(drain@(10,5) source@(10,55))")

# --- final plot via qimpy's own plotter ---
fv.update_stash(steps, tsim)
ckpt = os.path.join(tmp, "rect.h5")
with Checkpoint(ckpt, writable=True, rotate=False) as cp:
    fv._save_checkpoint(CheckpointPath(cp, "geometry"), CheckpointContext(""))
out = os.path.join(tmp, "rect_fluid_{}.png")
run_finite_volume([ckpt], slice(None), out, {"field": "n"},
                  {"density": 1.6, "linewidth": 0.9, "arrowsize": 0.9}, 130)
pngs = sorted([p for p in os.listdir(tmp) if p.endswith(".png")]); print("PNGs:", pngs)
shutil.copy(os.path.join(tmp, pngs[-1]), "/tmp/rect_fluid_final.png"); print("saved /tmp/rect_fluid_final.png")
