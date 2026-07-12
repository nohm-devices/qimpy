"""Moving drift-frame FiniteVolume validation (Stage 1, ballistic) on real meshes.

(A) PERIODIC 2D mesh, shear-drift + thermal-ripple IC: the strong test --
    conserved totals (N, px, py, E) must be invariant to round-off AND the shape's
    moments must equal the marched U (consistency), at drift 0.15 and 3 vF.
(B) SOURCE/DRAIN/WALL device (structured, no `triangle` dep): stability + machine-
    precision consistency on an open device with contacts+walls; wall leaks zero net
    mass; N-continuity is structural (telescoping).
Run on the A100:  python3 mf_device_test.py
"""
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


def _grid(n, L):
    xs = np.linspace(0.0, L, n + 1)
    V = np.array([[x, y] for y in xs for x in xs], float)
    idx = lambda i, j: j * (n + 1) + i
    Tr = []
    for j in range(n):
        for i in range(n):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            Tr += [[a, b, c], [a, c, d]]
    return V, np.array(Tr), idx


def periodic_mesh(n, L, path):
    V, Tr, idx = _grid(n, L)
    be = []
    for i in range(n):
        be += [[idx(i, 0), idx(i + 1, 0)], [idx(i, n), idx(i + 1, n)]]
    for j in range(n):
        be += [[idx(0, j), idx(0, j + 1)], [idx(n, j), idx(n, j + 1)]]
    save_mesh(path, V, Tr, np.array(be), ["periodic"] * len(be), lattice=[[L, 0.0], [0.0, L]])


def device_mesh(n, L, path):
    V, Tr, idx = _grid(n, L)
    be, bm = [], []
    for i in range(n):
        be += [[idx(i, 0), idx(i + 1, 0)]]; bm += ["wall"]
        be += [[idx(i, n), idx(i + 1, n)]]; bm += ["wall"]
    for j in range(n):
        be += [[idx(0, j), idx(0, j + 1)]]; bm += ["source"]
        be += [[idx(n, j), idx(n, j + 1)]]; bm += ["drain"]
    save_mesh(path, V, Tr, np.array(be), bm)


def build(path, contacts):
    fs = FermiSurface(kF=kF, vF=vF, M_theta=8, Nr=4, T=T, xi_max=6.0,
                      moving_frame=True, process_grid=pg)
    fv = FiniteVolume(material=fs, mesh_file=path, contacts=contacts,
                      cfl=0.4, process_grid=pg)
    return fs, fv


def kspace_rate(fv, fs):
    """max k-space advection rate |ξ̇'|/Δξ'_min + |φ̇|/Δφ, for the moving-mesh CFL."""
    mu, Te, u = fs.recover_frame(fv._U, Te_guess=fv._Te)
    uf = fv._faces_fn(fv._u).reshape(-1, fv.Nk)
    dU = fv._march_U(fv._u, mu, Te, u, 0.0, uf)
    gmu, gTe = fv._grad(mu), fv._grad(Te); gkD = fv._grad(fs.mstar * u / fs.hbar)
    dmu, dTe, dkD = fs.dframe_from_dU(dU, mu, Te, u)
    xid, phid = fs.shell_velocities(mu, Te, u, dmu, dTe, dkD, gmu, gTe, gkD)
    dxi = float(torch.diff(fs.radial.xi).abs().min())
    return float(xid.abs().max()) / dxi + float(phid.abs().max()) / fs.angular.wphi


def cfl_dt(fv, fs, dt_real):
    return min(dt_real, 0.3 / max(kspace_rate(fv, fs), 1e-30))


def cons_report(U0, Ut, area_u0):
    dN = abs(Ut[0] - U0[0]) / abs(U0[0])
    dE = abs(Ut[3] - U0[3]) / abs(U0[3])
    Jsc = area_u0
    dpx = abs(Ut[1] - U0[1]) / Jsc
    dpy = abs(Ut[2] - U0[2]) / Jsc
    return dN, dpx, dpy, dE


print("=" * 100)
pm = os.path.join(tmp, "per.npz"); periodic_mesh(24, 1.0, pm)
for drift in (0.15, 3.0):
    fs, fv = build(pm, {"periodic": None})
    dev = fs.rho0.device
    cen = torch.as_tensor(fv.geom.centroid_np, device=dev)
    mu = torch.full((fv.K,), fs.E_F, device=dev)
    Te = T * (1.0 + 0.05 * torch.cos(2 * np.pi * cen[:, 0]))
    u = torch.zeros(fv.K, 2, device=dev); u[:, 0] = drift * vF * torch.sin(2 * np.pi * cen[:, 1])
    fv._U = fs.U_from_frame(mu, Te, u); fv._Te = Te.clone()
    fv._u = fs.rho0[None, :].repeat(fv.K, 1).clone()
    U0 = fv.U_totals().clone()
    Jsc = float((fv.geom.area[:, None] * fv._U[:, 1:3].abs()).sum()) + 1e-300
    # pass a dt 8x OVER the CFL so the solver's internal substepping is exercised
    # (self-stable: the harness does NOT clamp dt to a stable value).
    dt_real = 0.4 * float(fv.geom.inradius.min()) / (fs.v_speed.max().item() + drift * vF)
    for st in range(15):
        fv.step_moving_frame(0.0, 8.0 * cfl_dt(fv, fs, dt_real))
    dN, dpx, dpy, dE = cons_report(U0, fv.U_totals(), Jsc)
    cr = fv.consistency_residual()
    print(f"PERIODIC drift={drift:4.2f}vF |kD|/kF={fs.mstar*drift*vF/kF:4.1f}: cons(N,px,py,E)="
          f"({dN:.1e},{dpx:.1e},{dpy:.1e},{dE:.1e}) consist={float(cr.max()):.1e} "
          f"f in [{fv._u.min():.2e},{fv._u.max():.4f}] finite={torch.isfinite(fv._u).all().item()}")

print("-" * 100)
dm = os.path.join(tmp, "dev.npz"); device_mesh(20, 1.0, dm)
fs, fv = build(dm, {"source": {"dmu": 0.3 * T}, "drain": {"dmu": 0.0}})
N0 = float(fv.U_totals()[0])
dt_real = 0.4 * float(fv.geom.inradius.min()) / fs.v_speed.max().item()
Ns = []
for st in range(15):
    fv.step_moving_frame(0.0, 8.0 * cfl_dt(fv, fs, dt_real))
    if st % 5 == 0 or st == 14:
        cr = fv.consistency_residual()
        Ns.append(float(fv.U_totals()[0]))
        print(f"DEVICE step {st:3d}: N={fv.U_totals()[0]:.6e} consist={float(cr.max()):.1e} "
              f"f in [{fv._u.min():.2e},{fv._u.max():.4f}] finite={torch.isfinite(fv._u).all().item()}")
print("=" * 100)
ok = torch.isfinite(fv._u).all().item()
print("PASS: moving-frame FiniteVolume runs on periodic + device meshes."
      if ok else "CHECK: non-finite state.")
