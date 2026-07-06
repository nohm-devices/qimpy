"""Tests for the cell-centered finite-volume solver (:class:`FiniteVolume`).

Covers the geometry operators (least-squares gradient, periodic edge pairing),
conservation (closed-domain mass, conservative contact-current readout) and the
full contact parity with the DG solver (fixed-voltage, floating probe, current
source). Run directly (serial) or under pytest.
"""
from __future__ import annotations
import os
import tempfile
from collections import Counter

import numpy as np
import torch

from qimpy import rc
from qimpy.mpi import ProcessGrid
from ..material import FermiSurface
from ._mesh import load_mesh, save_mesh
from ._finite_volume import FiniteVolume, build_fv_geom


# --------------------------------------------------------------------------- #
#  mesh generators (self-contained; qimpy does not mesh -- `triangle` is used
#  here only to produce small fixtures for the tests)
# --------------------------------------------------------------------------- #
def _make_rect_mesh(grid_spacing, path, all_walls=False):
    """rect-domain [5,105]x[5,55] with source/drain contact faces (mirrors
    examples/.../rect-domain.svg); all_walls=True closes it into a cavity."""
    import triangle as tr
    pts = np.array([[5, 5], [105, 5], [105, 55], [5, 55]], float)
    seg = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    m = tr.triangulate({"vertices": pts, "segments": seg},
                       f"pq30a{grid_spacing ** 2:g}")
    V, T = m["vertices"], m["triangles"]
    ec: Counter = Counter()
    for t in T:
        for x, y in [(0, 1), (1, 2), (2, 0)]:
            ec[tuple(sorted((int(t[x]), int(t[y]))))] += 1
    be = [e for e, c in ec.items() if c == 1]
    SRC, DRN = (10.0, 55.0, 5.0), (10.0, 5.0, 5.0)
    if all_walls:
        bm = ["wall"] * len(be)
    else:
        bm = []
        for a, b in be:
            mx, my = 0.5 * (V[a] + V[b])
            if (mx - SRC[0]) ** 2 + (my - SRC[1]) ** 2 <= SRC[2] ** 2:
                bm.append("source")
            elif (mx - DRN[0]) ** 2 + (my - DRN[1]) ** 2 <= DRN[2] ** 2:
                bm.append("drain")
            else:
                bm.append("wall")
    save_mesh(path, V, T, np.array(be), bm)
    return path


def _make_periodic_rect(n, L, path):
    """Structured n x n triangulation of [0,L]^2 with periodic lattice vectors."""
    xs = np.linspace(0.0, L, n + 1)
    V = np.array([[x, y] for y in xs for x in xs], float)

    def idx(i, j):
        return j * (n + 1) + i

    T, be = [], []
    for j in range(n):
        for i in range(n):
            a, b = idx(i, j), idx(i + 1, j)
            c, d = idx(i + 1, j + 1), idx(i, j + 1)
            T += [[a, b, c], [a, c, d]]
    for i in range(n):
        be += [[idx(i, 0), idx(i + 1, 0)], [idx(i, n), idx(i + 1, n)]]
    for j in range(n):
        be += [[idx(0, j), idx(0, j + 1)], [idx(n, j), idx(n, j + 1)]]
    save_mesh(path, V, np.array(T), np.array(be), ["periodic"] * len(be),
              lattice=[[L, 0.0], [0.0, L]])
    return path


def _make_strip_mesh(nx, ny, Lx, Ly, alpha_deg, path):
    """Tilted strip: [0,Lx]x[0,Ly] rotated by alpha so the top/bottom walls are
    OBLIQUE and the periodic lattice vector is (Lx cos a, Lx sin a) along the
    slant. At axis-aligned walls the discrete reflection coincides with a
    Galerkin operator by symmetry, masking the finite-N_theta tangential-
    quadrature artifact; at an oblique angle it does not, so this geometry
    discriminates the reflector's (D, T) tangential-momentum correction."""
    a = np.deg2rad(alpha_deg)
    c, s = np.cos(a), np.sin(a)
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    V = np.array([[x * c - y * s, x * s + y * c] for y in ys for x in xs], float)

    def idx(i, j):
        return j * (nx + 1) + i

    T = []
    for j in range(ny):
        for i in range(nx):
            a_, b_ = idx(i, j), idx(i + 1, j)
            c_, d_ = idx(i + 1, j + 1), idx(i, j + 1)
            T += [[a_, b_, c_], [a_, c_, d_]]
    be, bm = [], []
    for i in range(nx):
        be.append([idx(i, 0), idx(i + 1, 0)]); bm.append("wall")
        be.append([idx(i, ny), idx(i + 1, ny)]); bm.append("wall")
    for j in range(ny):
        be.append([idx(0, j), idx(0, j + 1)]); bm.append("periodic")
        be.append([idx(nx, j), idx(nx, j + 1)]); bm.append("periodic")
    save_mesh(path, V, np.array(T), np.array(be), bm, lattice=[[Lx * c, Lx * s]])
    return path


def _make_disk_mesh(R, n_seg, max_area, path, center=(50.0, 30.0)):
    """Triangulated disk: a circular boundary approximated by ``n_seg`` straight
    segments, every boundary edge a reflective wall. The boundary normals span
    all orientations, so it exercises the wall reflector at arbitrary angles."""
    import triangle as tr
    th = np.linspace(0.0, 2 * np.pi, n_seg, endpoint=False)
    pts = np.column_stack([center[0] + R * np.cos(th), center[1] + R * np.sin(th)])
    seg = np.column_stack([np.arange(n_seg), (np.arange(n_seg) + 1) % n_seg])
    m = tr.triangulate({"vertices": pts, "segments": seg}, f"pq30a{max_area:g}")
    V, T = m["vertices"], m["triangles"]
    ec: Counter = Counter()
    for t in T:
        for x, y in [(0, 1), (1, 2), (2, 0)]:
            ec[tuple(sorted((int(t[x]), int(t[y]))))] += 1
    be = [e for e, c in ec.items() if c == 1]
    save_mesh(path, V, T, np.array(be), ["wall"] * len(be))
    return path


# --------------------------------------------------------------------------- #
#  builders
# --------------------------------------------------------------------------- #
def _make_line_mesh(nx, path, Lx=1.0, ends=("source", "drain")):
    """1D line mesh: nx interval cells on [0, Lx] (y=0); the two ends are tagged
    ``ends`` (default source/drain)."""
    x = np.linspace(0.0, Lx, nx + 1)
    V = np.column_stack([x, np.zeros(nx + 1)])
    cells = np.column_stack([np.arange(nx), np.arange(1, nx + 1)])
    be = np.array([[0, 0], [nx, nx]], int)                # ends as degenerate (v,v)
    save_mesh(path, V, cells, be, list(ends))
    return path


def _build_fv(contacts, *, mesh_path=None, gs=12.0, vF=1.5, M=8, **mat_kw):
    """FermiSurface(Nr=1) device on a triangle mesh, wrapped in a FiniteVolume geometry."""
    tmp = tempfile.mkdtemp()
    path = mesh_path or _make_rect_mesh(gs, os.path.join(tmp, "rect.npz"))
    pg = ProcessGrid(rc.comm, "rk", (1, 1))
    kw = dict(kF=1.0, vF=vF, M_theta=M, Nr=1, T=1.0,
              tau_p=np.inf, tau_ee=np.inf, r_c=np.inf, specularity=1.0)
    kw.update(mat_kw)
    material = FermiSurface(process_grid=pg, **kw)
    geom = FiniteVolume(material=material, mesh_file=path, contacts=contacts,
                 process_grid=pg)
    return geom, material


def _step(geom, nsteps):
    """RK4 advance (advection + collisions, both inside rho_dot)."""
    dt = 0.5 * geom.dt_max
    for _ in range(nsteps):
        r0 = geom.rho
        k1 = geom.rho_dot(r0, 0.0)
        k2 = geom.rho_dot(r0 + (0.5 * dt) * k1, 0.0)
        k3 = geom.rho_dot(r0 + (0.5 * dt) * k2, 0.0)
        k4 = geom.rho_dot(r0 + dt * k3, 0.0)
        geom.rho = r0 + (dt / 6.0) * (k1 + 2 * (k2 + k3) + k4)


def _mass_rate(geom):
    """d/dt of total particle number = sum_k area_k * sum_c ncoef_c (du/dt)_kc."""
    dudt = geom.rho_dot(geom.rho, 0.0)[0]
    return float((geom.geom.area[:, None] * geom._ncoef[None, :] * dudt).sum())


def _integral(geom, material, obs_idx, t=0.0):
    """Domain integral of observable `obs_idx` (0=n, 1=jx, 2=jy): sum_k area_k o_k."""
    obs = torch.einsum("oc,kc->ko", material.get_observables(t), geom._u)  # (K,3)
    return float((geom.geom.area * obs[:, obs_idx]).sum())


# --------------------------------------------------------------------------- #
#  geometry operators
# --------------------------------------------------------------------------- #
def test_lsq_gradient_is_exact_on_linear_fields() -> None:
    """The fused reconstruction operator reproduces any linear field's face
    increments to machine precision -- so the scheme is exactly 2nd-order in the
    unlimited (smooth) regime, on a distorted/irregular mesh."""
    torch.set_default_dtype(torch.float64)
    tmp = tempfile.mkdtemp()
    g = build_fv_geom(load_mesh(_make_rect_mesh(9.0, os.path.join(tmp, "r.npz"))))
    dev = g.recon.device
    grad = torch.tensor([0.37, -1.21], dtype=torch.float64, device=dev)
    cen = torch.from_numpy(g.centroid_np).to(dev)
    u = (cen @ grad)[:, None]                            # (K, 1) linear field
    d = torch.einsum("kfg,kgc->kfc", g.recon, u[g.nbr] - u[:, None])  # (K,3,1)
    d_exact = torch.einsum("kfx,x->kf", _face_offsets(g).to(dev), grad)
    assert float((d[..., 0] - d_exact).abs().max()) < 1e-11


def _face_offsets(g):
    """(centroid -> face-midpoint) offset per cell/face, from the stored mesh."""
    p = torch.from_numpy(g.vertices_np)[torch.from_numpy(g.triangles_np)]  # (K,3,2)
    fmid = 0.5 * (p[:, [0, 1, 2]] + p[:, [1, 2, 0]])
    return fmid - p.mean(1)[:, None]


def test_periodic_lattice_promotes_all_boundary_edges() -> None:
    """On a fully periodic square every boundary edge pairs through the lattice
    and becomes interior, so there are no boundary edges and an arbitrary state
    conserves mass exactly (no faces can leak)."""
    torch.set_default_dtype(torch.float64)
    geom, _ = _build_fv({}, mesh_path=_periodic_mesh(6, 10.0))
    assert geom.geom.bcell.numel() == 0
    geom._u = torch.randn(geom.K, geom.Nk, device=rc.device)
    assert abs(_mass_rate(geom)) < 1e-12


def _periodic_mesh(n, L):
    tmp = tempfile.mkdtemp()
    return _make_periodic_rect(n, L, os.path.join(tmp, "per.npz"))


# --------------------------------------------------------------------------- #
#  conservation
# --------------------------------------------------------------------------- #
def test_closed_domain_conserves_mass() -> None:
    """A fully reflective (all-walls) cavity neither gains nor loses particles:
    the mass-conserving reflector drives the total mass rate to ~0."""
    torch.set_default_dtype(torch.float64)
    tmp = tempfile.mkdtemp()
    path = _make_rect_mesh(12.0, os.path.join(tmp, "rect.npz"), all_walls=True)
    geom, _ = _build_fv({}, mesh_path=path)
    geom._u = torch.randn(geom.K, geom.Nk, device=rc.device)   # arbitrary state
    assert abs(_mass_rate(geom)) < 1e-9


def test_contact_current_readout_is_conservative() -> None:
    """Sum of contact currents equals minus the total mass rate (walls carry no
    current), so the readout exactly accounts for the device's charge balance."""
    torch.set_default_dtype(torch.float64)
    geom, _ = _build_fv({"source": {"dmu": 0.1}, "drain": {"dmu": -0.1}})
    _step(geom, 40)
    I = geom.contact_currents(0.0)
    assert abs(sum(I.values()) + _mass_rate(geom)) < 1e-9


# --------------------------------------------------------------------------- #
#  contact parity
# --------------------------------------------------------------------------- #
def test_floating_contact_carries_no_current() -> None:
    """A voltage probe's level floats to zero its own current, exactly."""
    torch.set_default_dtype(torch.float64)
    geom, _ = _build_fv({"source": {"dmu": 0.1}, "drain": {"floating": True}})
    _step(geom, 150)
    assert abs(geom.contact_currents(0.0)["drain"]) < 1e-12


def test_floating_contact_reads_uniform_potential() -> None:
    """A floating probe in a device held at a uniform genuine Fermi-Dirac level
    mu0 (isotropic f_r = sigmoid(mu0 - xi_r)) reads back mu0.  An isotropic state
    carries zero net current per radial node (sum_q v.n = 0 over the angular
    ordinates), so the floating Newton -- which solves sum_r sigmoid(mu-xi_r) B_r
    = -C_out -- returns exactly the seed's chemical potential.  (Under full f a
    uniform occupancy f = V0 would instead read logit(V0), not V0: 'level' is a
    genuine chemical potential, not a delta-f amplitude.)"""
    torch.set_default_dtype(torch.float64)
    geom, mat = _build_fv({"source": {"dmu": 0.1}, "drain": {"floating": True}})
    xi_r = mat.radial.xi.to(rc.device)                   # (Nr,)
    N_theta = mat.angular.N_theta
    for mu0 in (0.05, -0.1, 0.2):
        f_iso = torch.sigmoid(mu0 - xi_r).repeat_interleave(N_theta)   # (Nk,)
        geom._u = f_iso[None, :].repeat(geom.K, 1)
        geom.contact_currents(0.0)                       # solves the feedback level
        assert abs(geom.contact_potentials()["drain"] - mu0) < 1e-9


def test_current_source_zero_equals_floating() -> None:
    """A current source with I_set = 0 reproduces a floating probe exactly."""
    torch.set_default_dtype(torch.float64)
    geom, _ = _build_fv({"source": {"dmu": 0.1}, "drain": {"I_set": 0.0}})
    _step(geom, 150)
    assert abs(geom.contact_currents(0.0)["drain"]) < 1e-12


def test_current_source_delivers_prescribed_current() -> None:
    """Each evaluation a current source self-adjusts its level so the net outward
    flux equals I_set; the same-flux readout then agrees to roundoff every step.
    Two sources drive current through a resistive device."""
    torch.set_default_dtype(torch.float64)
    I_target = 0.05
    geom, _ = _build_fv(
        {"source": {"I_set": -I_target}, "drain": {"I_set": +I_target}},
        tau_p=15.0, tau_ee=8.0)
    _step(geom, 40)
    I = geom.contact_currents(0.0)
    assert abs(I["source"] + I_target) < 1e-10, I["source"]
    assert abs(I["drain"] - I_target) < 1e-10, I["drain"]
    V = geom.contact_potentials()
    assert V["source"] > V["drain"]                      # injector sits at higher mu


def test_current_source_polarity_reverses_with_sign() -> None:
    """Flipping the sign of I_set swaps the device potentials."""
    torch.set_default_dtype(torch.float64)
    gp, _ = _build_fv({"source": {"I_set": -0.02}, "drain": {"I_set": +0.02}})
    gn, _ = _build_fv({"source": {"I_set": +0.02}, "drain": {"I_set": -0.02}})
    _step(gp, 40); _step(gn, 40)
    Ip, In = gp.contact_currents(0.0), gn.contact_currents(0.0)
    Vp, Vn = gp.contact_potentials(), gn.contact_potentials()
    assert abs(Ip["source"] + In["source"]) < 1e-10
    assert (Vp["source"] - Vp["drain"]) * (Vn["source"] - Vn["drain"]) < 0


# --------------------------------------------------------------------------- #
#  long-time wall physics (specular reflection conserves mass + tangential mom.)
# --------------------------------------------------------------------------- #
def _steps_for(geom, t_end):
    return int(t_end / (0.5 * geom.dt_max))


def test_reflective_walls_conserve_mass_long_time() -> None:
    """A closed (all-wall) cavity conserves total particle number to ~machine
    precision as a density blob traverses the domain and actively reflects --
    the time-integrated companion to the instantaneous-rate test above."""
    torch.set_default_dtype(torch.float64)
    tmp = tempfile.mkdtemp()
    path = _make_rect_mesh(10.0, os.path.join(tmp, "rect.npz"), all_walls=True)
    geom, mat = _build_fv({}, mesh_path=path)
    cen = torch.from_numpy(geom.geom.centroid_np).to(rc.device)
    q0 = torch.tensor([55.0, 30.0], dtype=torch.float64, device=rc.device)
    blob = torch.exp(-((cen - q0) ** 2).sum(-1) / (2 * 6.0 ** 2))   # (K,)
    geom._u = blob[:, None].repeat(1, geom.Nk)                      # isotropic = density
    m0 = _integral(geom, mat, 0)
    _step(geom, _steps_for(geom, 30.0))
    assert abs(_integral(geom, mat, 0) - m0) / abs(m0) < 1e-10


def test_oblique_wall_conserves_tangential_momentum() -> None:
    """On a tilted strip (oblique specular walls + periodic along the slant) the
    wall-tangent current J_tang = cos(a) jx + sin(a) jy is a global invariant of
    specular reflection. Holds to round-off with the (D,T) reflector; the older
    single-D scheme drifts ~1e-4 here -- this is the discriminating test."""
    torch.set_default_dtype(torch.float64)
    alpha = 23.7                                       # oblique: not 0/45/90 deg
    a = np.deg2rad(alpha); ca, sa = float(np.cos(a)), float(np.sin(a))
    Lx, Ly = 40.0, 20.0
    tmp = tempfile.mkdtemp()
    mesh = _make_strip_mesh(8, 4, Lx, Ly, alpha, os.path.join(tmp, "strip.npz"))
    geom, mat = _build_fv({}, mesh_path=mesh)
    cen = geom.geom.centroid_np
    d_perp = -sa * cen[:, 0] + ca * cen[:, 1] - 0.5 * Ly
    blob = np.exp(-(d_perp ** 2) / (2 * 2.0 ** 2))                  # (K,)
    theta = mat.angular.theta.detach().cpu().numpy()               # (Nk,)
    u0 = 1.0 * blob[:, None] + 0.3 * np.cos(theta - a)[None, :]     # density + drift
    geom._u = torch.as_tensor(u0, device=rc.device, dtype=torch.float64)
    n0 = _integral(geom, mat, 0)
    J0 = ca * _integral(geom, mat, 1) + sa * _integral(geom, mat, 2)
    _step(geom, _steps_for(geom, 20.0))
    n1 = _integral(geom, mat, 0)
    J1 = ca * _integral(geom, mat, 1) + sa * _integral(geom, mat, 2)
    assert abs(n1 - n0) / abs(n0) < 1e-10, f"mass drift {(n1 - n0) / n0:.2e}"
    assert abs(J1 - J0) / abs(J0) < 1e-10, f"J_tang drift {(J1 - J0) / J0:.2e}"


# --------------------------------------------------------------------------- #
#  contact-driven steady state
# --------------------------------------------------------------------------- #
def test_contact_driven_state_is_bounded() -> None:
    """Source/drain contacts (dmu = +/-0.1 about the half-filled f0 = 0.5) drive a
    finite, Pauli-admissible full-f solution: f neither blows up nor leaves [0, 1]
    (the genuine-FD contact ghosts stay in range and the streaming is monotone),
    up to tiny MUSCL overshoots.  The delta-f 'interior density < contact range'
    bound no longer applies -- the full-f density carries the ~0.5 background."""
    torch.set_default_dtype(torch.float64)
    geom, mat = _build_fv({"source": {"dmu": 0.1}, "drain": {"dmu": -0.1}})
    _step(geom, _steps_for(geom, 50.0))
    f = geom._u
    assert torch.isfinite(f).all(), "contact-driven solution diverged"
    assert float(f.min()) > -1e-2, float(f.min())
    assert float(f.max()) < 1.0 + 1e-2, float(f.max())


def test_biased_contacts_balance_at_steady_state() -> None:
    """In a resistive device (finite tau) the source and drain currents relax to
    equal and opposite as the device approaches DC steady state, with a real
    current flowing. The exact-balance residual is -d/dt(mass), which decays on
    the device's (slow) charging time, so the resolution-independent invariant
    checked here is the *relative* imbalance |I_s + I_d| / |I_s|. Stepped with
    collisions, since a ballistic cavity rings rather than settling."""
    torch.set_default_dtype(torch.float64)
    geom, _ = _build_fv({"source": {"dmu": 0.1}, "drain": {"dmu": -0.1}},
                        tau_p=15.0, tau_ee=8.0)
    _step(geom, _steps_for(geom, 600.0))
    I = geom.contact_currents(0.0)
    assert abs(I["source"]) > 1e-3, "no current flowing"
    assert abs(I["source"] + I["drain"]) / abs(I["source"]) < 1e-2, I  # equal & opp.


def test_curved_mass_conservation() -> None:
    """A closed disk -- a curved (circular) boundary approximated by straight wall
    edges whose normals span all orientations -- conserves total mass to ~machine
    precision as a blob expands and reflects. The FV flux form is conservative and
    the reflector zeroes net mass flux on every edge regardless of its angle, so
    cell-centered FV needs no isoparametric/arc projectors (unlike the high-order
    DG version this replaces, which was skipped for that reason)."""
    torch.set_default_dtype(torch.float64)
    tmp = tempfile.mkdtemp()
    path = _make_disk_mesh(20.0, 64, 4.0, os.path.join(tmp, "disk.npz"))
    geom, mat = _build_fv({}, mesh_path=path)
    cen = torch.from_numpy(geom.geom.centroid_np).to(rc.device)
    q0 = torch.tensor([50.0, 30.0], dtype=torch.float64, device=rc.device)
    blob = torch.exp(-((cen - q0) ** 2).sum(-1) / (2 * 5.0 ** 2))
    geom._u = blob[:, None].repeat(1, geom.Nk)
    m0 = _integral(geom, mat, 0)
    _step(geom, _steps_for(geom, 30.0))
    assert abs(_integral(geom, mat, 0) - m0) / abs(m0) < 1e-10


# --------------------------------------------------------------------------- #
#  full-f streaming (evolve the FULL distribution f in [0,1], NOT delta-f).
#
#  The interior FV upwind is LINEAR in f, so the committed interior scheme
#  streams full f exactly; these tests exercise the NONLINEAR full-f boundary
#  conditions (genuine Fermi-Dirac walls / reservoirs) and the (D,T) specular
#  correction applied on top of the isotropic equilibrium f0.  The current-
#  source test is the direct answer to "inject a known current with vD = 0".
# --------------------------------------------------------------------------- #
def _f0_full(mat):
    """Isotropic equilibrium f0(xi_r) the full-f runs seed to (nodal, (Nk,))."""
    return mat.rho0.to(rc.device)


def _drift_ang(mat, amp, phase=0.0):
    """``amp cos(theta_q - phase)`` replicated across all radial nodes -> (Nk,).

    Channel layout is c = r*N_theta + q (radial outer, angular inner), so the
    per-ordinate angular pattern tiles Nr times."""
    ang = amp * torch.cos(mat.angular.theta - phase)          # (N_theta,)
    return ang.repeat(mat.Nr).to(rc.device)                   # (Nr*N_theta,)


def test_full_f_equilibrium_is_stationary() -> None:
    """The isotropic Fermi-Dirac equilibrium f0 is an EXACT stationary state of
    full-f streaming in a closed cavity: div(v f0) = 0 cell-by-cell (a closed
    polygon has sum (v.n) len = 0), the specular reflector returns f0 (isotropic
    -> zero (D,T) correction), and the collisionless m=0 mode does not decay.
    So rho_dot(f0) vanishes to round-off -- nothing spurious drives the vacuum."""
    torch.set_default_dtype(torch.float64)
    tmp = tempfile.mkdtemp()
    path = _make_rect_mesh(12.0, os.path.join(tmp, "rect.npz"), all_walls=True)
    geom, mat = _build_fv({}, mesh_path=path)
    geom._u = _f0_full(mat)[None, :].repeat(geom.K, 1)
    rate = geom.rho_dot(geom.rho, 0.0)[0]
    assert float(rate.abs().max()) < 1e-10, float(rate.abs().max())


def test_full_f_closed_box_conserves_mass() -> None:
    """A closed (all-wall) cavity conserves total particle number as a full-f
    state (f0 + density blob + angular drift) streams and specularly reflects:
    the (D,T)-corrected reflector zeroes net normal mass flux on every edge."""
    torch.set_default_dtype(torch.float64)
    tmp = tempfile.mkdtemp()
    path = _make_rect_mesh(11.0, os.path.join(tmp, "rect.npz"), all_walls=True)
    geom, mat = _build_fv({}, mesh_path=path)
    cen = torch.from_numpy(geom.geom.centroid_np).to(rc.device)
    q0 = torch.tensor([55.0, 30.0], dtype=torch.float64, device=rc.device)
    blob = 0.05 * torch.exp(-((cen - q0) ** 2).sum(-1) / (2 * 8.0 ** 2))   # (K,)
    geom._u = (_f0_full(mat)[None, :] + blob[:, None]
               + _drift_ang(mat, 0.03)[None, :])
    m0 = _integral(geom, mat, 0)
    _step(geom, _steps_for(geom, 25.0))
    assert abs(_integral(geom, mat, 0) - m0) / abs(m0) < 1e-10


def test_full_f_closed_box_conserves_mass_multiradial() -> None:
    """Nr=4 (four energy nodes) full-f state conserves mass in a closed
    AXIS-ALIGNED box.  At axis-aligned walls the specular reflection theta ->
    pi-theta is an exact angular-node permutation for every radial node, so mass
    is conserved per radial mode regardless of the vF-weighted (D,T) correction
    (which is only vF-exact, not v_speed-exact, at oblique walls for Nr>1)."""
    torch.set_default_dtype(torch.float64)
    tmp = tempfile.mkdtemp()
    path = _make_rect_mesh(13.0, os.path.join(tmp, "rect.npz"), all_walls=True)
    geom, mat = _build_fv({}, mesh_path=path, Nr=4)
    cen = torch.from_numpy(geom.geom.centroid_np).to(rc.device)
    q0 = torch.tensor([55.0, 30.0], dtype=torch.float64, device=rc.device)
    blob = 0.05 * torch.exp(-((cen - q0) ** 2).sum(-1) / (2 * 8.0 ** 2))
    geom._u = (_f0_full(mat)[None, :] + blob[:, None]
               + _drift_ang(mat, 0.03)[None, :])
    m0 = _integral(geom, mat, 0)
    _step(geom, _steps_for(geom, 20.0))
    assert abs(_integral(geom, mat, 0) - m0) / abs(m0) < 1e-10


def test_full_f_specular_oblique_conserves_tangential_momentum() -> None:
    """Full-f specular reflection at an OBLIQUE wall conserves both mass and the
    wall-tangent momentum J_tang = cos(a) jx + sin(a) jy to round-off.  f0 is
    isotropic (carries zero net normal and tangential flux), so the (D,T)
    correction computed on full f equals that on delta-f; this is the
    discriminating oblique test for the full-f reflector (Nr=1: vF = v_speed)."""
    torch.set_default_dtype(torch.float64)
    alpha = 23.7                                       # oblique: not 0/45/90 deg
    a = np.deg2rad(alpha); ca, sa = float(np.cos(a)), float(np.sin(a))
    Lx, Ly = 40.0, 20.0
    tmp = tempfile.mkdtemp()
    mesh = _make_strip_mesh(8, 4, Lx, Ly, alpha, os.path.join(tmp, "strip.npz"))
    geom, mat = _build_fv({}, mesh_path=mesh)
    cen = geom.geom.centroid_np
    d_perp = -sa * cen[:, 0] + ca * cen[:, 1] - 0.5 * Ly
    blob = np.exp(-(d_perp ** 2) / (2 * 2.0 ** 2))                  # (K,)
    theta = mat.angular.theta.detach().cpu().numpy()               # (Nk,) at Nr=1
    f0 = _f0_full(mat).detach().cpu().numpy()                      # (Nk,)
    u0 = f0[None, :] + 0.05 * blob[:, None] + 0.3 * np.cos(theta - a)[None, :]
    geom._u = torch.as_tensor(u0, device=rc.device, dtype=torch.float64)
    n0 = _integral(geom, mat, 0)
    J0 = ca * _integral(geom, mat, 1) + sa * _integral(geom, mat, 2)
    _step(geom, _steps_for(geom, 20.0))
    n1 = _integral(geom, mat, 0)
    J1 = ca * _integral(geom, mat, 1) + sa * _integral(geom, mat, 2)
    assert abs(n1 - n0) / abs(n0) < 1e-10, f"mass drift {(n1 - n0) / n0:.2e}"
    assert abs(J1 - J0) / abs(J0) < 1e-9, f"J_tang drift {(J1 - J0) / J0:.2e}"


def test_full_f_specular_curved_conserves_mass() -> None:
    """Full-f specular reflection conserves mass on a closed disk (curved wall,
    normals spanning all angles): the (D,T) correction zeroes net normal mass
    flux on every straight wall edge regardless of its orientation (Nr=1)."""
    torch.set_default_dtype(torch.float64)
    tmp = tempfile.mkdtemp()
    path = _make_disk_mesh(20.0, 64, 4.0, os.path.join(tmp, "disk.npz"))
    geom, mat = _build_fv({}, mesh_path=path)
    cen = torch.from_numpy(geom.geom.centroid_np).to(rc.device)
    q0 = torch.tensor([50.0, 30.0], dtype=torch.float64, device=rc.device)
    blob = 0.05 * torch.exp(-((cen - q0) ** 2).sum(-1) / (2 * 5.0 ** 2))
    geom._u = _f0_full(mat)[None, :] + blob[:, None]               # isotropic bump
    m0 = _integral(geom, mat, 0)
    _step(geom, _steps_for(geom, 25.0))
    assert abs(_integral(geom, mat, 0) - m0) / abs(m0) < 1e-10


def test_full_f_diffuse_closed_box_conserves_mass() -> None:
    """A fully diffuse (specularity=0) closed box conserves mass: each wall face
    re-emits an isotropic genuine Fermi-Dirac at a single chemical potential,
    solved by a 1-D Newton for zero net normal mass flux (exact to Newton
    tolerance; the diffuse mass carrier is v_speed-weighted at any Nr)."""
    torch.set_default_dtype(torch.float64)
    tmp = tempfile.mkdtemp()
    path = _make_rect_mesh(12.0, os.path.join(tmp, "rect.npz"), all_walls=True)
    geom, mat = _build_fv({}, mesh_path=path, specularity=0.0)
    cen = torch.from_numpy(geom.geom.centroid_np).to(rc.device)
    q0 = torch.tensor([55.0, 30.0], dtype=torch.float64, device=rc.device)
    blob = 0.05 * torch.exp(-((cen - q0) ** 2).sum(-1) / (2 * 8.0 ** 2))
    geom._u = (_f0_full(mat)[None, :] + blob[:, None]
               + _drift_ang(mat, 0.03)[None, :])
    m0 = _integral(geom, mat, 0)
    _step(geom, _steps_for(geom, 20.0))
    assert abs(_integral(geom, mat, 0) - m0) / abs(m0) < 1e-9


def test_full_f_diffuse_drains_tangential_momentum() -> None:
    """Contrast on the oblique strip: full-f SPECULAR walls conserve the tangent
    momentum J_tang, while full-f DIFFUSE walls (isotropic FD re-emit carries
    zero tangential momentum) drain it.  Both conserve mass.  Confirms the
    diffuse wall is a genuine tangential-momentum sink for full f, not delta-f
    only -- the physics the {specular, diffuse} split is supposed to capture."""
    torch.set_default_dtype(torch.float64)
    alpha = 23.7
    a = np.deg2rad(alpha); ca, sa = float(np.cos(a)), float(np.sin(a))
    Lx, Ly = 40.0, 20.0
    tmp = tempfile.mkdtemp()
    mesh = _make_strip_mesh(8, 4, Lx, Ly, alpha, os.path.join(tmp, "strip.npz"))

    def run(spec):
        geom, mat = _build_fv({}, mesh_path=mesh, specularity=spec)
        cen = geom.geom.centroid_np
        d_perp = -sa * cen[:, 0] + ca * cen[:, 1] - 0.5 * Ly
        blob = np.exp(-(d_perp ** 2) / (2 * 2.0 ** 2))
        theta = mat.angular.theta.detach().cpu().numpy()
        f0 = _f0_full(mat).detach().cpu().numpy()
        u0 = f0[None, :] + 0.05 * blob[:, None] + 0.3 * np.cos(theta - a)[None, :]
        geom._u = torch.as_tensor(u0, device=rc.device, dtype=torch.float64)
        n0 = _integral(geom, mat, 0)
        J0 = ca * _integral(geom, mat, 1) + sa * _integral(geom, mat, 2)
        _step(geom, _steps_for(geom, 40.0))
        n1 = _integral(geom, mat, 0)
        J1 = ca * _integral(geom, mat, 1) + sa * _integral(geom, mat, 2)
        return n0, n1, J0, J1

    n0s, n1s, J0s, J1s = run(1.0)                      # specular
    n0d, n1d, J0d, J1d = run(0.0)                      # diffuse
    assert abs(n1s - n0s) / abs(n0s) < 1e-9, "specular mass"
    assert abs(n1d - n0d) / abs(n0d) < 1e-9, "diffuse mass"
    assert abs(J1s - J0s) / abs(J0s) < 1e-9, \
        f"specular should conserve J_tang: {(J1s - J0s) / J0s:.2e}"
    assert abs(J1d) < 0.9 * abs(J0d), \
        f"diffuse should drain J_tang: {J1d:.3e} vs {J0d:.3e}"


def test_full_f_diffuse_curved_multiradial_conserves_mass() -> None:
    """Nr=4 fully-diffuse disk conserves mass: the diffuse re-emit balances the
    v_speed-weighted normal mass flux at every radial node and every wall angle,
    so curved + multiradial mass is machine-precise (the diffuse mass carrier is
    exact for any Nr, unlike the vF-weighted specular (D,T) correction)."""
    torch.set_default_dtype(torch.float64)
    tmp = tempfile.mkdtemp()
    path = _make_disk_mesh(20.0, 64, 5.0, os.path.join(tmp, "disk.npz"))
    geom, mat = _build_fv({}, mesh_path=path, specularity=0.0, Nr=4)
    cen = torch.from_numpy(geom.geom.centroid_np).to(rc.device)
    q0 = torch.tensor([50.0, 30.0], dtype=torch.float64, device=rc.device)
    blob = 0.05 * torch.exp(-((cen - q0) ** 2).sum(-1) / (2 * 5.0 ** 2))
    geom._u = _f0_full(mat)[None, :] + blob[:, None]
    m0 = _integral(geom, mat, 0)
    _step(geom, _steps_for(geom, 20.0))
    assert abs(_integral(geom, mat, 0) - m0) / abs(m0) < 1e-9


def test_full_f_current_source_injects_known_current() -> None:
    """Inject a KNOWN current with a genuine-FD reservoir at vD = 0 -- the direct
    answer to "inject a known current by setting a dmu and vD = 0".  A {I_set,
    vD=0} current source samples an ISOTROPIC Fermi-Dirac whose level (its
    chemical potential dmu) the geometry solves each step (1-D Newton) so the net
    emitted current equals I_set.  The same-measure readout then reproduces I_set
    to round-off, and the injector floats to the higher potential.  Stepped with
    collisions (resistive device) so a real current develops in the bulk."""
    torch.set_default_dtype(torch.float64)
    I = 0.03
    geom, mat = _build_fv(
        {"source": {"I_set": -I}, "drain": {"I_set": +I}},
        tau_p=15.0, tau_ee=8.0)
    _step(geom, 40)
    Ic = geom.contact_currents(0.0)
    assert abs(Ic["source"] + I) < 1e-9, Ic["source"]
    assert abs(Ic["drain"] - I) < 1e-9, Ic["drain"]
    V = geom.contact_potentials()
    assert V["source"] > V["drain"]                    # injector sits at higher mu
    obs = torch.einsum("oc,kc->ko", mat.get_observables(0.0), geom._u)
    jmag = torch.sqrt(obs[:, 1] ** 2 + obs[:, 2] ** 2)
    assert float(jmag.max()) > 1e-4, "no real current flowing in the bulk"


def test_full_f_pauli_bounds_preserved() -> None:
    """Full-f streaming keeps the distribution Pauli-admissible: genuine-FD
    contact ghosts (dmu = +/-3 -> f in ~[0.047, 0.953]) are in [0,1] and the
    isotropic f0 background never leaves [0,1], so after driving to a bounded
    state f stays within [0,1] up to tiny MUSCL overshoots (no hard positivity
    clamp is applied on the FV path -- the genuine-FD BCs keep it in range)."""
    torch.set_default_dtype(torch.float64)
    geom, mat = _build_fv({"source": {"dmu": 3.0}, "drain": {"dmu": -3.0}})
    _step(geom, _steps_for(geom, 50.0))
    f = geom._u
    assert torch.isfinite(f).all(), "full-f solution diverged"
    assert float(f.min()) > -1e-2, float(f.min())
    assert float(f.max()) < 1.0 + 1e-2, float(f.max())


# --------------------------------------------------------------------------- #
#  spatial decomposition (bit-for-bit vs serial). Runs as two subprocesses of
#  this module in "worker" mode (FV_MPI_OUT set) -- one serial, one mpirun -n 2.
# --------------------------------------------------------------------------- #
def _decomp_worker() -> None:
    """Step the rect problem and save the final state in input (un-permuted)
    cell order; invoked as a subprocess by test_decomp_matches_serial. Builds an
    auto-sized process grid (r split over ranks, k=1) so it works at any rank
    count, unlike the fixed (1,1) grid the serial-test builder uses."""
    rc.init()
    torch.set_default_dtype(torch.float64)
    pg = ProcessGrid(rc.comm, "rk", None)
    pg.provide_n_tasks("k", 1)
    mat = FermiSurface(kF=1.0, vF=1.5, M_theta=8, Nr=1, T=1.0,
                       tau_p=15.0, tau_ee=8.0, r_c=np.inf, specularity=1.0,
                       process_grid=pg)
    geom = FiniteVolume(material=mat, mesh_file=os.environ["FV_MPI_MESH"],
                  contacts={"source": {"dmu": 0.1}, "drain": {"floating": True}},
                  process_grid=pg)
    cen = torch.from_numpy(geom.geom.centroid_np).to(rc.device)
    geom._u = torch.zeros(geom.K, geom.Nk, device=rc.device)
    geom._u[:, 0] = 0.01 * (1.0 + cen[:, 0] / 100.0 + cen[:, 1] / 50.0)  # partition-invariant
    dt = 0.5 * geom.dt_max
    for _ in range(30):
        r0 = geom.rho
        geom.rho = r0 + dt * geom.rho_dot(r0 + 0.5 * dt * geom.rho_dot(r0, 0.0), 0.0)
    owned = geom._u[geom._own_start:geom._own_stop].detach().cpu().numpy()
    parts = rc.comm.gather(owned, root=0)
    if rc.comm.rank == 0:
        full = np.concatenate(parts, axis=0)            # renumbered order
        u = np.empty_like(full)
        if geom._perm is not None:
            u[geom._perm] = full                        # back to input order
        else:
            u = full
        np.save(os.environ["FV_MPI_OUT"], u)


def test_1d_line_mesh_ballistic_is_antisymmetric() -> None:
    """A 1D wire (interval cells, 2 faces/cell) with source/drain dmu=+/-0.1 runs
    stably through the FiniteVolume 1D geometry path.  Under full f the ballistic
    steady state obeys the particle-hole + parity symmetry S: x->L-x, v->-v,
    f->1-f -- at Nr=1 this maps the source FD ghost sigmoid(0.1-xi) exactly onto
    the drain's (1 - sigmoid(-0.1-xi) = sigmoid(0.1-xi)).  So the density
    DEVIATION from half-filling is antisymmetric, n(x)+n(L-x) = 2 n_eq, while the
    current is spatially uniform (the +/-x populations carry it straight through
    and the isotropic f0 background carries none)."""
    torch.set_default_dtype(torch.float64)
    tmp = tempfile.mkdtemp()
    mesh = _make_line_mesh(40, os.path.join(tmp, "line.npz"))
    geom, mat = _build_fv({"source": {"dmu": 0.1}, "drain": {"dmu": -0.1}},
                          mesh_path=mesh)
    assert geom._nf == 2                                   # interval cells -> 2 faces
    _step(geom, _steps_for(geom, 20.0))
    x = geom.geom.centroid_np[:, 0]
    obs = torch.einsum("oc,kc->ko", mat.get_observables(0.0), geom._u)
    n = obs[:, 0].cpu().numpy()
    jx = obs[:, 1].cpu().numpy()
    assert np.isfinite(n).all(), "1D solution diverged"
    n_eq = float((mat.get_observables(0.0)[0] * _f0_full(mat)).sum())   # half-filled
    dev = n - n_eq
    mirror = np.array([int(np.argmin(np.abs(x - (1.0 - xi)))) for xi in x])
    assert np.linalg.norm(dev + dev[mirror]) / (np.linalg.norm(dev) + 1e-30) < 1e-9
    assert abs(jx.mean()) > 1e-3, "no ballistic current"
    assert jx.std() / abs(jx.mean()) < 1e-2, "ballistic current not uniform"


def test_decomp_matches_serial() -> None:
    """The METIS spatial decomposition reproduces the serial solve bit-for-bit:
    a 2-rank run (partition + 2-ring halo exchange) equals the 1-rank run on the
    same problem to round-off. Spawns this module in worker mode (serial, then
    mpirun -n 2) and compares; needs mpirun + pymetis."""
    import subprocess
    import sys
    tmp = tempfile.mkdtemp()
    mesh = _make_rect_mesh(12.0, os.path.join(tmp, "rect.npz"))
    mod = "qimpy.transport.geometry.test_finite_volume"
    f1, f2 = os.path.join(tmp, "u1.npy"), os.path.join(tmp, "u2.npy")
    env = dict(os.environ, FV_MPI_MESH=mesh)
    subprocess.run([sys.executable, "-m", mod], check=True, env=dict(env, FV_MPI_OUT=f1))
    subprocess.run(["mpirun", "-n", "2", sys.executable, "-m", mod], check=True,
                   env=dict(env, FV_MPI_OUT=f2))
    u1, u2 = np.load(f1), np.load(f2)
    assert np.allclose(u1, u2, atol=1e-12, rtol=0), float(np.abs(u1 - u2).max())


if __name__ == "__main__":
    if os.environ.get("FV_MPI_OUT"):           # subprocess worker for the test above
        _decomp_worker()
        raise SystemExit
    rc.init()
    test_lsq_gradient_is_exact_on_linear_fields(); print("lsq_gradient_exact: PASS")
    test_periodic_lattice_promotes_all_boundary_edges(); print("periodic_promote: PASS")
    test_closed_domain_conserves_mass(); print("closed_domain_mass: PASS")
    test_contact_current_readout_is_conservative(); print("contact_readout_conservative: PASS")
    test_floating_contact_carries_no_current(); print("floating_zero_current: PASS")
    test_floating_contact_reads_uniform_potential(); print("floating_reads_potential: PASS")
    test_current_source_zero_equals_floating(); print("current_source_zero: PASS")
    test_current_source_delivers_prescribed_current(); print("current_source_delivers: PASS")
    test_current_source_polarity_reverses_with_sign(); print("current_source_polarity: PASS")
    test_reflective_walls_conserve_mass_long_time(); print("reflective_mass_long_time: PASS")
    test_oblique_wall_conserves_tangential_momentum(); print("oblique_wall_tang_momentum: PASS")
    test_contact_driven_state_is_bounded(); print("contact_driven_bounded: PASS")
    test_biased_contacts_balance_at_steady_state(); print("biased_balance: PASS")
    test_curved_mass_conservation(); print("curved_mass_conservation: PASS")
    test_full_f_equilibrium_is_stationary(); print("full_f_equilibrium_stationary: PASS")
    test_full_f_closed_box_conserves_mass(); print("full_f_closed_box_mass: PASS")
    test_full_f_closed_box_conserves_mass_multiradial(); print("full_f_closed_box_mass_Nr4: PASS")
    test_full_f_specular_oblique_conserves_tangential_momentum(); print("full_f_specular_oblique_tang: PASS")
    test_full_f_specular_curved_conserves_mass(); print("full_f_specular_curved_mass: PASS")
    test_full_f_diffuse_closed_box_conserves_mass(); print("full_f_diffuse_closed_box_mass: PASS")
    test_full_f_diffuse_drains_tangential_momentum(); print("full_f_diffuse_drains_tang: PASS")
    test_full_f_diffuse_curved_multiradial_conserves_mass(); print("full_f_diffuse_curved_mass_Nr4: PASS")
    test_full_f_current_source_injects_known_current(); print("full_f_current_source: PASS")
    test_full_f_pauli_bounds_preserved(); print("full_f_pauli_bounds: PASS")
    test_decomp_matches_serial(); print("decomp_matches_serial: PASS")
    print("ALL PASS")
