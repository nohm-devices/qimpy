"""2nd-order Venkatakrishnan HLLC, tau_p=inf, mu-pinning contact: save frames (n,Jx,Jy)
for a detailed density+current movie.  Frames + mesh dumped to /tmp for sandbox rendering."""
import numpy as np, torch, tempfile, os
from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.transport.material import FermiSurface
from qimpy.transport.geometry._finite_volume import FiniteVolume
from qimpy.transport.geometry._mesh import save_mesh
torch.set_default_dtype(torch.float64)
pg = ProcessGrid(rc.comm, "rk", (1, 1)); tmp = tempfile.mkdtemp()

X0, X1, Y0, Y1 = 5., 105., 5., 55.; SRC, DRN = (10., 55., 5.), (10., 5., 5.); h = 3.0
ny = max(2, 2 * round(0.5 * (Y1 - Y0) / h)); nx = 2 * ny
xs = np.linspace(X0, X1, nx + 1); ys = np.linspace(Y0, Y1, ny + 1)
Vl = [[x, y] for y in ys for x in xs]; vid = lambda i, j: j * (nx + 1) + i; T = []
for j in range(ny):
    for i in range(nx):
        a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1); cen = len(Vl)
        Vl.append([0.5 * (xs[i] + xs[i + 1]), 0.5 * (ys[j] + ys[j + 1])])
        T += [[a, b, cen], [b, c, cen], [c, d, cen], [d, a, cen]]
Vv = np.array(Vl, float); Tn = np.array(T, int)
edges = ([(vid(i, 0), vid(i + 1, 0)) for i in range(nx)] + [(vid(i, ny), vid(i + 1, ny)) for i in range(nx)] +
         [(vid(0, j), vid(0, j + 1)) for j in range(ny)] + [(vid(nx, j), vid(nx, j + 1)) for j in range(ny)])
be, bm = [], []
for a, b in edges:
    mx, my = 0.5 * (Vv[a] + Vv[b])
    bm.append("source" if (mx - SRC[0]) ** 2 + (my - SRC[1]) ** 2 <= SRC[2] ** 2 else
              "drain" if (mx - DRN[0]) ** 2 + (my - DRN[1]) ** 2 <= DRN[2] ** 2 else "wall"); be.append([a, b])
dm = os.path.join(tmp, "rect.npz"); save_mesh(dm, Vv, Tn, np.array(be, int), bm)

fs = FermiSurface(kF=1.0, vF=1.5, M_theta=12, Nr=1, T=0.05, xi_max=6.0, fluid_model=True, process_grid=pg)
fs._fluid_use_hllc = True            # HLLC macroscopic Riemann
fs._fluid_first_order = False        # 2nd-order Venkatakrishnan reconstruction
fs._fluid_dirichlet_contact = False  # transparent contact (bounded; validate BJ vs Venkat churn)
fs._bj_limiter = True                # Barth-Jespersen strict limiter (THE FIX)
fs.tau_inv_p = 0.0                   # tau_p = inf (no drag)
fv = FiniteVolume(material=fs, mesh_file=dm,
                  contacts={"source": {"dmu": 0.1}, "drain": {"dmu": -0.1}}, cfl=0.4, process_grid=pg)
mu0, Te0, _ = fs.recover_frame(fv._U)
dt = 0.4 * float(fv.geom.inradius.min()) / float(fs.sound_speed(mu0, Te0).max())
cen = np.asarray(fv.geom.centroid_np)
np.savez("/tmp/movie_mesh.npz", V=Vv, T=Tn, cx=cen[:, 0], cy=cen[:, 1])

NSTEPS = 4000; SAVE_EVERY = 20       # up to ~200 frames
frames = []; tsim = 0.
def dump():
    ts = np.array([f[0] for f in frames])
    ns = np.stack([f[1] for f in frames]); jx = np.stack([f[2] for f in frames]); jy = np.stack([f[3] for f in frames])
    np.savez_compressed("/tmp/movie_data.npz", t=ts, n=ns, jx=jx, jy=jy)
for st in range(NSTEPS + 1):
    if st % SAVE_EVERY == 0:
        U = fv._U.detach().cpu().numpy()
        fin = bool(np.isfinite(U).all())
        frames.append((tsim, U[:, 0].copy(), U[:, 1].copy(), U[:, 2].copy()))
        mu, Te, u = fs.recover_frame(fv._U)
        Ma = float(u.norm(dim=1).max()) / float(fs.sound_speed(mu, Te).max())
        print(f"frame {len(frames):3d} st={st:5d} t={tsim:7.1f} Mach={Ma:.3f} "
              f"n/n0[{U[:,0].min()/U[:,0].mean():.3f},{U[:,0].max()/U[:,0].mean():.3f}] fin={fin}", flush=True)
        if not fin:
            print("*** non-finite -> stop ***", flush=True); break
        if len(frames) % 15 == 0:
            dump()
    if st < NSTEPS:
        fv.step_fluid(tsim, dt); tsim += dt
dump()
print(f"saved /tmp/movie_data.npz  {len(frames)} frames", flush=True)
