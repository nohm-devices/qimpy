"""PROVE the drain answer: same clean HLLC 1st-order scheme, two contact models.
  transparent (FD-reservoir split): drains current, does NOT pin density -> drain ~ -1%.
  dirichlet  (mu-pinning HLLC):     pins n(E_F+dmu) -> drain shows the full -dmu/E_F ~ -13%.
Reports contact-cell densities + domain range; saves a density+current plot for the Dirichlet case."""
import numpy as np, torch, tempfile, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
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


def run(dirichlet, nblocks=120, plot=None, tau_inv=0.0):
    fs = FermiSurface(kF=1.0, vF=1.5, M_theta=12, Nr=1, T=0.05, xi_max=6.0, fluid_model=True, process_grid=pg)
    fs._fluid_use_hllc = True; fs._fluid_first_order = True; fs._fluid_dirichlet_contact = dirichlet
    fs.tau_inv_p = tau_inv
    fv = FiniteVolume(material=fs, mesh_file=dm,
                      contacts={"source": {"dmu": 0.1}, "drain": {"dmu": -0.1}}, cfl=0.4, process_grid=pg)
    mu0, Te0, _ = fs.recover_frame(fv._U)
    dt = 0.4 * float(fv.geom.inradius.min()) / float(fs.sound_speed(mu0, Te0).max())
    cen = torch.as_tensor(fv.geom.centroid_np, device=fv._U.device); n0 = float(fv._U[:, 0].mean())
    nres_s = float(fs.n_FD(torch.tensor(fs.E_F + 0.1), torch.tensor(fs.T_temp))) / n0
    nres_d = float(fs.n_FD(torch.tensor(fs.E_F - 0.1), torch.tensor(fs.T_temp))) / n0
    i_s = int(((cen[:, 0] - 10.) ** 2 + (cen[:, 1] - 55.) ** 2).argmin())
    i_d = int(((cen[:, 0] - 10.) ** 2 + (cen[:, 1] - 5.) ** 2).argmin())
    tag0 = "DIRICHLET" if dirichlet else "transparent"
    print(f"\n[{tag0}] reservoir targets src {nres_s:.3f} drn {nres_d:.3f}", flush=True)
    prev = fv._U.clone(); tsim = 0.
    for blk in range(nblocks):
        for _ in range(50):
            fv.step_fluid(tsim, dt); tsim += dt
        ch = float((fv._U - prev).abs().max()) / (float(fv._U.abs().max()) + 1e-300)
        nb = fv._U[:, 0]; fin = bool(torch.isfinite(fv._U).all())
        if blk % 4 == 0 or blk == nblocks - 1 or not fin:
            mu_, Te_, u_ = fs.recover_frame(fv._U)
            Ma = float(u_.norm(dim=1).max()) / float(fs.sound_speed(mu_, Te_).max())
            print(f"  blk{blk:3d} rel={ch:.2e} Mach={Ma:.3f} contact src={float(nb[i_s])/n0:.4f} "
                  f"drn={float(nb[i_d])/n0:.4f} range[{float(nb.min())/n0:.4f},{float(nb.max())/n0:.4f}] fin={fin}", flush=True)
        if not fin:
            print(f"  [{tag0}] non-finite @ blk {blk}"); return
        if ch < 3e-6:
            print(f"  [{tag0}] STEADY @ blk {blk}"); break
        prev = fv._U.clone()
    n = fv._U[:, 0]; Jx, Jy = fv._U[:, 1], fv._U[:, 2]
    imn = int(n.argmin()); imx = int(n.argmax())
    tag = "DIRICHLET (mu-pinning)" if dirichlet else "transparent (FD-reservoir)"
    print(f"\n[{tag}]  stopped blk{blk} rel={ch:.1e}", flush=True)
    print(f"  reservoir targets: source {nres_s:.3f}  drain {nres_d:.3f}", flush=True)
    print(f"  contact cells:     source n/n0={float(n[i_s])/n0:.4f}  drain n/n0={float(n[i_d])/n0:.4f}", flush=True)
    print(f"  domain n/n0 range [{float(n.min())/n0:.4f},{float(n.max())/n0:.4f}]  "
          f"min@(x={float(cen[imn,0]):.0f},y={float(cen[imn,1]):.0f}) max@(x={float(cen[imx,0]):.0f},y={float(cen[imx,1]):.0f})", flush=True)
    mu, Te, u = fs.recover_frame(fv._U)
    print(f"  drift Mach max={float(u.norm(dim=1).max())/float(fs.sound_speed(mu,Te).max()):.3f}  "
          f"mean Jy y>30={float(Jy[cen[:,1]>30].mean()):+.3e} y<30={float(Jy[cen[:,1]<30].mean()):+.3e}", flush=True)
    if plot:
        tri = Triangulation(Vv[:, 0], Vv[:, 1], Tn); nn = len(Vv); w = fv.geom.area.cpu().numpy()
        nnp = n.cpu().numpy(); drho = (nnp - nnp.mean()) / n0
        def node(cv):
            num = np.zeros(nn); den = np.zeros(nn)
            for k in range(3):
                idx = Tn[:, k]; np.add.at(num, idx, cv * w); np.add.at(den, idx, w)
            return num / np.maximum(den, 1e-30)
        gxg = np.linspace(X0 + .3, X1 - .3, 260); gyg = np.linspace(Y0 + .3, Y1 - .3, 130)
        GX, GY = np.meshgrid(gxg, gyg)
        Ui = LinearTriInterpolator(tri, node(Jx.cpu().numpy()))(GX, GY).filled(0.)
        Vi = LinearTriInterpolator(tri, node(Jy.cpu().numpy()))(GX, GY).filled(0.)
        fig, ax = plt.subplots(figsize=(12.4, 6.9)); dmax = max(abs(drho).max(), 1e-9)
        tpc = ax.tripcolor(tri, facecolors=drho, cmap="RdBu_r", vmin=-dmax, vmax=dmax, shading="flat")
        sp = np.hypot(Ui, Vi)
        ax.streamplot(gxg, gyg, Ui, Vi, color="k", density=1.7,
                      linewidth=0.3 + 2.4 * np.minimum(sp / (sp.max() + 1e-30), 1.), arrowsize=0.8)
        ax.plot([SRC[0]-SRC[2], SRC[0]+SRC[2]], [Y1, Y1], color="red", lw=6, solid_capstyle="butt")
        ax.plot([DRN[0]-DRN[2], DRN[0]+DRN[2]], [Y0, Y0], color="blue", lw=6, solid_capstyle="butt")
        cb = fig.colorbar(tpc, ax=ax, shrink=.85, pad=.01); cb.set_label(r"$(n-\langle n\rangle)/n_0$")
        ax.set_aspect("equal"); ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"qimpy fluid, HLLC 1st-order, mu-PINNING (Dirichlet) contact -- "
                     f"drain underdensity {100*(float(n[i_d])/n0-1):.0f}%", fontsize=10)
        fig.tight_layout(); fig.savefig(plot, dpi=120); print(f"  wrote {plot}", flush=True)


run(dirichlet=True, nblocks=80, plot="/tmp/rect_dirichlet.png", tau_inv=0.1)
run(dirichlet=False, nblocks=30)
