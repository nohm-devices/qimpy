"""A/B on the rect setup: interior flux = HLLC vs own-frame KFVS split (same mesh, same
_face_frames reconstruction, same FD-reservoir contacts, tau_p=inf).  Reports per-block
boundedness + source->drain density dipole + density-min location.  No plot (numbers only)."""
import numpy as np, torch, tempfile, os
from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.transport.material import FermiSurface
from qimpy.transport.geometry._finite_volume import FiniteVolume
from qimpy.transport.geometry._mesh import save_mesh
torch.set_default_dtype(torch.float64)
pg = ProcessGrid(rc.comm, "rk", (1, 1))
tmp = tempfile.mkdtemp()

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
print(f"mesh: {len(T)} tris  src={bm.count('source')} drn={bm.count('drain')} wall={bm.count('wall')}", flush=True)


def run_mode(use_hllc, nblocks=80, nsteps=50):
    fs = FermiSurface(kF=1.0, vF=1.5, M_theta=12, Nr=1, T=0.05, xi_max=6.0, fluid_model=True, process_grid=pg)
    fs._fluid_use_hllc = use_hllc
    fv = FiniteVolume(material=fs, mesh_file=dm,
                      contacts={"source": {"dmu": 0.1}, "drain": {"dmu": -0.1}}, cfl=0.4, process_grid=pg)
    mu0, Te0, _ = fs.recover_frame(fv._U)
    dt = 0.4 * float(fv.geom.inradius.min()) / float(fs.sound_speed(mu0, Te0).max())
    cen = torch.as_tensor(fv.geom.centroid_np, device=fv._U.device)
    src_m = ((cen[:, 0] - 10.) ** 2 + (cen[:, 1] - 55.) ** 2) < 15. ** 2
    drn_m = ((cen[:, 0] - 10.) ** 2 + (cen[:, 1] - 5.) ** 2) < 15. ** 2
    print(f"\n=== interior flux = {'HLLC' if use_hllc else 'KFVS-split'}  (dt={dt:.4f}) ===", flush=True)
    prev = fv._U.clone(); tsim = 0.; blown = None
    for blk in range(nblocks):
        for _ in range(nsteps):
            fv.step_fluid(tsim, dt); tsim += dt
        fin = bool(torch.isfinite(fv._U).all().item())
        mu, Te, u = fs.recover_frame(fv._U); n = fv._U[:, 0]
        ch = float((fv._U - prev).abs().max()) / (float(fv._U.abs().max()) + 1e-300)
        ngrad = (float(n[src_m].mean()) - float(n[drn_m].mean())) / float(n.mean())
        imn = int(n.argmin()); umax = float(u.norm(dim=1).max()) / 1.5
        if blk % 5 == 0 or (not fin) or umax > 2:
            print(f" blk{blk:3d} rel={ch:.2e} umax/vF={umax:.3e} (n_src-n_drn)/n={ngrad:+.3e} "
                  f"nmin@(x={float(cen[imn,0]):.0f},y={float(cen[imn,1]):.0f})={float(n[imn]):.4f} fin={fin}", flush=True)
        if (not fin) or umax > 5:
            blown = blk; print(f"  *** BLEW UP at block {blk} ***", flush=True); break
        prev = fv._U.clone()
    n = fv._U[:, 0]; imn = int(n.argmin()); imx = int(n.argmax())
    if blown is None:
        print(f"  FINAL: (n_src-n_drn)/n={(float(n[src_m].mean())-float(n[drn_m].mean()))/float(n.mean()):+.3e}  "
              f"nmin@(x={float(cen[imn,0]):.0f},y={float(cen[imn,1]):.0f})={float(n[imn]):.4f}  "
              f"nmax@(x={float(cen[imx,0]):.0f},y={float(cen[imx,1]):.0f})={float(n[imx]):.4f}", flush=True)
    return blown


b_hllc = run_mode(True)
b_kfvs = run_mode(False)
print("\n" + "=" * 52, flush=True)
print(f"HLLC interior : {'BLEW at blk '+str(b_hllc) if b_hllc is not None else 'BOUNDED'}", flush=True)
print(f"KFVS interior : {'BLEW at blk '+str(b_kfvs) if b_kfvs is not None else 'BOUNDED'}", flush=True)
print("=" * 52, flush=True)
