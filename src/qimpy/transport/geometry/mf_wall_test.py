"""Moving-measure wall boundary operator: EXACT conservation of the physically-correct
moments at a reflecting wall.  An ALL-WALL box (no contacts) is seeded with a drifted+heated
state (drift has a wall-NORMAL component, so the KFVS moving measure a=v.n^+u.n^ differs from
the reflector's no-drift v.n^ measure -- the whole point) and run at tau_p=inf.

A wall passes ZERO net normal MASS flux for ANY specularity -> total N conserved EXACTLY.
SPECULAR (s=1) reflection is elastic + free-slip: each wall edge additionally carries ZERO
tangential-momentum flux and ZERO energy flux, so total ENERGY is conserved EXACTLY and the
walls inject ZERO net tangential momentum.  DIFFUSE (s<1) balances only mass; its tangential
drag and energy thermalization are PHYSICAL and must NOT vanish.

Reports, per specularity:
  |dN|/N           total-mass drift                       (all s: ~1e-14)
  |dE|/E           total-energy drift                     (s=1: ~1e-13; s<1: physical, nonzero)
  |Pt_inj|/|Pn_inj|  net wall tangential- vs normal-momentum injection over the run
                     (s=1: ~1e-13; s<1: physical drag, nonzero)
  max_edge tang/norm max per-edge tangential- vs normal-momentum flux ratio over the run
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
    bn = fv.geom.bn.to(dev); blen = fv.geom.blen.to(dev)           # (Nb,2),(Nb,)
    nxw, nyw = bn[:, 0], bn[:, 1]
    is_w = fv._is_wall_b.to(dev)                                   # all True for the box
    N0 = float((area * fv._U[:, 0]).sum())
    E0 = float((area * fv._U[:, 3]).sum())
    dt = 0.4 * float(fv.geom.inradius.min()) / (fs.v_speed.max().item() + 0.05 * vF)
    Pt_inj = 0.0; Pn_inj = 0.0; max_ratio = 0.0
    for st in range(60):
        fv.step_moving_frame(0.0, dt)
        fJ = fv._Ff_bnd_J                                          # (Nb,2) last-substep wall Jx.n^,Jy.n^
        tang = torch.where(is_w, (-fJ[:, 0] * nyw + fJ[:, 1] * nxw) * blen, torch.zeros_like(blen))
        norm = torch.where(is_w, (fJ[:, 0] * nxw + fJ[:, 1] * nyw) * blen, torch.zeros_like(blen))
        Pt_inj += dt * float(tang.sum())                          # net tangential momentum injected
        Pn_inj += dt * float(norm.sum())                          # net normal momentum (wall pressure)
        denom = float(norm.abs().max()) + 1e-300
        max_ratio = max(max_ratio, float(tang.abs().max()) / denom)   # per-edge tang/norm
    N1 = float((area * fv._U[:, 0]).sum())
    E1 = float((area * fv._U[:, 3]).sum())
    fin = torch.isfinite(fv._U).all().item()
    dN = abs(N1 - N0) / abs(N0)
    dE = abs(E1 - E0) / abs(E0)
    ptn = abs(Pt_inj) / (abs(Pn_inj) + 1e-300)
    print(f"\nspecularity={s:.2f}  finite={fin}")
    print(f"  |dN|/N             = {dN:.3e}")
    print(f"  |dE|/E             = {dE:.3e}")
    print(f"  |Pt_inj|/|Pn_inj|  = {ptn:.3e}   (net wall tangential vs normal momentum)")
    print(f"  max_edge tang/norm = {max_ratio:.3e}")
    assert fin, f"non-finite state at specularity={s}"
    assert dN < 1e-14, f"mass NOT conserved at specularity={s}: |dN|/N={dN:.3e}"
    if s >= 1.0:
        # Specular = elastic free-slip: energy AND tangential momentum conserved exactly.
        assert dE < 1e-13, f"specular energy NOT conserved: |dE|/E={dE:.3e}"
        assert ptn < 1e-13, f"specular tangential momentum NOT conserved: {ptn:.3e}"
        assert max_ratio < 1e-13, f"specular per-edge tangential flux NOT zero: {max_ratio:.3e}"
    # s<1: dE and the tangential ratios are the PHYSICAL diffuse drag/heat -- not asserted zero.

print("\nOK: walls conserve mass (any s) AND tangential momentum + energy (specular) exactly.")
