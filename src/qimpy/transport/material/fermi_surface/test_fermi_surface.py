from __future__ import annotations
import os

import numpy as np
import torch
import pytest

from qimpy import rc
from qimpy.mpi import get_process_grid
from qimpy.transport.material import FermiSurface
from ._fermi_surface import AngularBasis, RadialBasis
from ._representation import _DeltaKReflector

pytestmark = pytest.mark.mpi_skip  # All tests in this module are serial


def _make(
    n_theta: int,
    n_r: int = 1,
    *,
    T: float = 1.0,
    tau_p: float = np.inf,
    tau_ee: float = np.inf,
    r_c: float = np.inf,
    specularity: float = 1.0,
) -> FermiSurface:
    return FermiSurface(
        kF=1.0,
        vF=1.5,
        n_theta=n_theta,
        n_r=n_r,
        T=T,
        tau_p=tau_p,
        tau_ee=tau_ee,
        r_c=r_c,
        specularity=specularity,
        process_grid=get_process_grid("rk", (1, 1)),
    )


# ------ Basis transformations -------


@pytest.mark.parametrize("n_theta", [8, 16])
def test_angular_basis_roundtrip(n_theta: int) -> None:
    basis = AngularBasis(n_theta)
    eye = basis.to_modes @ basis.from_modes
    assert torch.allclose(eye, torch.eye(n_theta, dtype=torch.float64), atol=1e-12)


@pytest.mark.parametrize("n_r", [2, 4])
@pytest.mark.parametrize("T", [0.5, 2.0])
def test_radial_basis_orthonormal(n_r: int, T: float) -> None:
    basis = RadialBasis(n_r, T=T)
    eye = basis.to_modes @ basis.from_modes
    assert torch.allclose(eye, torch.eye(n_r, dtype=torch.float64), atol=1e-10)


@pytest.mark.parametrize("n_r", [1, 2])
@pytest.mark.parametrize("n_theta", [8])
def test_transform_roundtrip(n_r: int, n_theta: int) -> None:
    fs = _make(n_theta=n_theta, n_r=n_r)
    n_modes = n_r * n_theta
    a = torch.randn(4, n_modes, dtype=torch.float64, device=rc.device)
    f = fs.from_modes(a)
    assert torch.allclose(a, fs.to_modes(fs.from_modes(a)), atol=1e-12)
    assert torch.allclose(f, fs.from_modes(fs.to_modes(f)), atol=1e-12)


def test_constant_at_mode0() -> None:
    fs = _make(n_theta=8, n_r=1)
    f = torch.full((1, fs.n_theta), 0.7, dtype=torch.float64, device=rc.device)
    a = fs.to_modes(f).flatten()
    assert a[0].abs() > 1e-6
    assert a[1:].abs().max() < 1e-12


@pytest.mark.parametrize(
    "tau_p, tau_ee, r_c",
    [
        (np.inf, np.inf, np.inf),  # ballistic, no field
        (0.7, np.inf, np.inf),  # tau_p only
        (0.7, 1.3, np.inf),  # tau_p + tau_ee
        (np.inf, np.inf, 2.0),  # cyclotron only
        (0.7, 1.3, 2.0),  # full
    ],
)
def test_rho_dot_matches_ref(tau_p: float, tau_ee: float, r_c: float) -> None:
    fs = _make(n_theta=16, n_r=1, tau_p=tau_p, tau_ee=tau_ee, r_c=r_c)
    # Hand-built modal rates: m=0 conserved, m=1 -> 1/tau_p, m>=2 -> 1/tau_p + 1/tau_ee
    rates = np.zeros(fs.n_theta)
    rate_p = 0.0 if not np.isfinite(tau_p) else 1.0 / tau_p
    rate_ee = 0.0 if not np.isfinite(tau_ee) else 1.0 / tau_ee
    rates[2:] = rate_p + rate_ee
    rates[1] = rates[fs.angular.m_max + 1] = rate_p
    a = torch.randn(4, fs.n_theta, dtype=torch.float64, device=rc.device)
    rho = fs.from_modes(a)
    rho_dot_fs = fs.rho_dot(rho, 0.0, 0)
    rho_dot_modal = -rc.from_np(rates) * a
    if fs.omega_c:
        da_dtheta = torch.einsum("dc, ...c -> ...d", fs.angular.derivative, a)
        rho_dot_modal += fs.omega_c * da_dtheta
    rho_dot_ref = fs.from_modes(rho_dot_modal)
    atol = 1.0e-12 * float(rho_dot_ref.abs().max())
    assert torch.allclose(rho_dot_fs, rho_dot_ref, atol=atol, rtol=0.0)


def test_contactor_linear() -> None:
    """Linear contactor applies dmu to m=0 and vD to m=1 alone."""
    fs = _make(n_theta=8, n_r=1)
    vD = 0.13
    dmu = 0.07
    # Wall normals at three angles
    phi = torch.tensor([0.3, 1.1, -2.4], dtype=torch.float64)
    n = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)
    contactor = fs.get_contactor(n, dmu=dmu, vD=vD)
    rho_dk = contactor(0.0)  # (3, N_theta)
    a = fs.to_modes(rho_dk)
    # Expected: a_0 = dmu; a_1 = -kF vD cos phi; b_1 = -kF vD sin phi
    m_max = fs.angular.m_max
    assert torch.allclose(a[:, 0], torch.full_like(a[:, 0], dmu), atol=1e-12)
    assert torch.allclose(a[:, 1], -(fs.kF * vD) * torch.cos(phi), atol=1e-12)
    assert torch.allclose(a[:, m_max + 1], -(fs.kF * vD) * torch.sin(phi), atol=1e-12)


def test_contactor_linear_limit() -> None:
    """Check that linear contact distribution matches linear limit of nonlinear one."""
    fs = _make(n_theta=16, n_r=1)
    phi = torch.tensor([0.4, -1.9], dtype=torch.float64)
    n = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)
    eps = 1e-6 * fs.T
    rho_linear = fs.get_contactor(n, dmu=eps, vD=eps)(0.0)
    rho_nonlinear = fs.get_contactor(n, dmu=eps, vD=eps, nonlinear=True)(0.0)
    scale = rho_linear.abs().max()
    assert (rho_nonlinear - rho_linear).abs().max() < 1e-6 * scale


def test_contactor_saturation() -> None:
    """Check that nonlinear contact captures saturation from Fermi function."""
    T = 0.02
    dmu = 0.1
    fs = _make(n_theta=8, n_r=1, T=T)
    n = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    rho_nonlinear = fs.get_contactor(n, dmu=dmu, nonlinear=True)(0.0)
    rho_ref = (2 * T) * np.tanh(dmu / (2 * T))
    assert torch.allclose(rho_nonlinear, torch.full_like(rho_nonlinear, rho_ref))
    rho_linear = fs.get_contactor(n, dmu=dmu)(0.0)
    assert rho_nonlinear.abs().max() < rho_linear.abs().max()  # saturation


def test_reflector_specular_axis_aligned() -> None:
    """x-axis wall specular reflection flips sine coefficients alone."""
    fs = _make(n_theta=8, n_r=1)
    n = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    refl = _DeltaKReflector(fs, n, specularity=1.0)
    a_in = torch.randn(1, fs.n_theta, dtype=torch.float64)
    u_in = fs.from_modes(a_in)
    a_out = fs.to_modes(refl(u_in))
    expected = a_in.clone()
    expected[..., fs.angular.m_max + 1 :] *= -1.0
    assert torch.allclose(a_out, expected, atol=1e-12)


@pytest.mark.parametrize("phi_deg", [23.7, 45.0, 90.0])
@pytest.mark.parametrize("s", [0.0, 0.7])
def test_reflector_conserves_charge(phi_deg: float, s: float) -> None:
    fs = _make(n_theta=8, n_r=1, specularity=s)
    phi = np.deg2rad(phi_deg)
    n = torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=torch.float64)
    refl = _DeltaKReflector(fs, n, specularity=s)
    rng = torch.Generator(device=rc.device).manual_seed(0)
    u_in = torch.randn(1, 1, fs.n_theta, dtype=torch.float64, generator=rng)
    u_out = refl(u_in)
    v_dot_n = fs.vF * (
        n[0, 0] * torch.cos(fs.angular.theta) + n[0, 1] * torch.sin(fs.angular.theta)
    )
    out_pos = v_dot_n.clamp(min=0)
    out_neg = v_dot_n.clamp(max=0)
    F_out = (out_pos * u_in[0, 0]).sum()
    F_in = (out_neg * u_out[0, 0]).sum()
    net = float((F_out + F_in).abs())
    scale = max(float(F_out.abs()), 1e-30)
    assert net / scale < 1e-12


@pytest.mark.parametrize("phi_deg", [23.7, 45.0, 90.0])
@pytest.mark.parametrize("specularity", [0.0, 0.7])
def test_reflector_tangemtial_momentum(phi_deg: float, specularity) -> None:
    """Specular reflection should conserve tangential momentum,
    while diffuse reflection should scale it by (1 - specularity)."""
    fs = _make(n_theta=8, n_r=1, specularity=specularity)
    phi = np.deg2rad(phi_deg)
    n = torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=torch.float64)
    refl = _DeltaKReflector(fs, n, specularity=specularity)
    rng = torch.Generator(device=rc.device).manual_seed(0)
    u_in = torch.randn(1, 1, fs.n_theta, dtype=torch.float64, generator=rng)
    u_out = refl(u_in)
    theta = fs.angular.theta
    v_dot_n = fs.vF * (n[0, 0] * torch.cos(theta) + n[0, 1] * torch.sin(theta))
    v_tang = fs.vF * (-n[0, 1] * torch.cos(theta) + n[0, 0] * torch.sin(theta))
    out_pos = v_dot_n.clamp(min=0)
    out_neg = v_dot_n.clamp(max=0)
    F_out_tang = (out_pos * v_tang * u_in[0, 0]).sum()
    F_in_tang = (out_neg * v_tang * u_out[0, 0]).sum()
    F_total = F_out_tang + F_in_tang
    expected = (1.0 - specularity) * F_out_tang  # continuum identity
    scale = max(float(F_out_tang.abs()), 1e-30)
    err = float((F_total - expected).abs()) / scale
    assert err < 1e-12


@pytest.mark.parametrize("phi_deg", [0.0, 23.7, 45.0])
@pytest.mark.parametrize("s", [0.0, 1.0])
def test_reflector_energy_conservation(phi_deg: float, s: float) -> None:
    fs = _make(n_theta=8, n_r=1, specularity=s)
    phi = np.deg2rad(phi_deg)
    n = torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=torch.float64)
    refl = _DeltaKReflector(fs, n, specularity=s)
    rng = torch.Generator(device=rc.device).manual_seed(0)
    u_in = torch.randn(1, 1, fs.n_theta, dtype=torch.float64, generator=rng)
    u_out = refl(u_in)
    theta = fs.angular.theta
    v_dot_n = fs.vF * (n[0, 0] * torch.cos(theta) + n[0, 1] * torch.sin(theta))
    out_pos = v_dot_n.clamp(min=0)
    out_neg = v_dot_n.clamp(max=0)
    energy = 0.5 * fs.vF**2  # constant on Fermi circle
    F_out = (out_pos * energy * u_in[0, 0]).sum()
    F_in = (out_neg * energy * u_out[0, 0]).sum()
    net = float((F_out + F_in).abs())
    scale = max(float(F_out.abs()), 1e-30)
    assert net / scale < 1e-12


def _make_cart(
    n_r: int, T: float = 0.05, *, local_te_rates: bool = True, ee=None
) -> FermiSurface:
    return FermiSurface(
        kF=1.0,
        vF=1.5,
        n_theta=6,
        n_r=n_r,
        T=T,
        xi_max=6.0,
        cartesian=dict(dk=T / (3.0 * 1.5), local_te_rates=local_te_rates),
        ee_scattering=ee,
        process_grid=get_process_grid("rk", (1, 1)),
    )


def _cart_state(fs, radial_mode: int, amp: float):
    """delta-f = w_eq_RB(xi_lab) * amp * psi_n(xi) * cos(2 theta): a pure
    (n, m=2 cos) mode of Phi about the lab frame (no N/J/E content)."""
    rep = fs.representation
    xi = (rep.eps_k - fs.mu) / fs.T
    th = torch.atan2(rep.k[:, 1], rep.k[:, 0])
    f0 = torch.special.expit(-xi)
    psi = rep._psi(xi)[:, radial_mode]
    return (f0 * (1 - f0) / fs.T) * amp * psi * torch.cos(2 * th)


@pytest.mark.parametrize("n_r, mode", [(1, 0), (2, 0), (2, 1)])
def test_cartesian_projection_contract(n_r, mode: int) -> None:
    """The coefficients handed to the modal operator ARE the Phi coefficients
    (Gram-corrected, T-normalized): a pure (n, m=2) input of amplitude amp
    projects to amp on that channel and ~0 elsewhere."""
    fs = _make_cart(n_r)
    amp = 1e-8
    rho = _cart_state(fs, mode, amp)[None]  # one spatial cell
    seen = {}

    def capture(a, te=None):
        seen["a"] = a.detach().clone()
        return torch.zeros_like(a)

    fs.representation.apply_collision(rho, capture)
    a = seen["a"].reshape(n_r, fs.n_theta)
    idx = (mode, 2)  # (n, m=2 cos)
    err_chan = abs(float(a[idx]) - amp) / amp
    others = a.clone()
    others[idx] = 0.0
    assert err_chan < 1e-4, f"channel amplitude off by {err_chan:.1e}"
    assert float(others.abs().max()) < 2e-2 * amp, "cross-channel leakage"


@pytest.mark.parametrize("Nr,mode", [(1, 0), (2, 1)])
def test_cartesian_projection_roundtrip(Nr: int, mode: int) -> None:
    fs = _make_cart(Nr)
    rho = _cart_state(fs, mode, 1e-8)[None]
    out = fs.representation.apply_collision(rho, lambda a, te=None: a)
    rel = float((out - rho).abs().max() / rho.abs().max())
    assert rel < 1e-4, f"P.R != identity: rel={rel:.1e}"


def _make_cart_closure(Nr=4, T=0.05, tau_ee=200.0):
    return FermiSurface(
        kF=1.0,
        vF=1.5,
        n_theta=6,
        n_r=Nr,
        T=T,
        xi_max=6.0,
        tau_ee=tau_ee,
        cartesian=dict(dk=T / (3.0 * 1.5), local_te_rates=False),
        residual_damping=True,
        process_grid=get_process_grid("rk", (1, 1)),
    )


def _drifted_heated(rep, fs, kD, Te, dmu=0.0):
    """Exact local drifted-heated FD as a deviation about f0_lab."""
    kp = rep.k - torch.as_tensor(kD, device=rep.k.device, dtype=rep.k.dtype)
    eps_p = kp.square().sum(-1) / (2 * fs.m_star)
    f_le = torch.special.expit(-(eps_p - (fs.mu + dmu)) / Te)
    return f_le - rep._f0_lab


def test_residual_closure_vanishes_on_local_equilibrium() -> None:
    """C[f_le] = 0 must survive the closure.  Checked at a NON-half-cell drift:
    at kD = 0 the parity reflection is an exact grid permutation, so a zero-
    drift test cannot see an interpolation defect."""
    fs_on = _make_cart_closure()
    fs_off = FermiSurface(
        kF=1.0,
        vF=1.5,
        n_theta=6,
        n_r=4,
        T=0.05,
        xi_max=6.0,
        tau_ee=200.0,
        cartesian=dict(dk=0.05 / 4.5, local_te_rates=False),
        process_grid=get_process_grid("rk", (1, 1)),
    )
    rep = fs_on.representation
    kD = 0.37 * rep._dk_grid * np.array([1.0, 0.61])
    df = _drifted_heated(rep, fs_on, kD, 1.3 * fs_on.T)[None]
    # ON vs OFF isolates the CLOSURE.  Comparing against zero instead would
    # measure the pre-existing frame-recovery / projection quadrature error,
    # which is finite on any grid and has nothing to do with this patch.
    a = rep.apply_collision(df.clone(), fs_on._modal_collision)
    b = fs_off.representation.apply_collision(df.clone(), fs_off._modal_collision)
    # Normalize by the ACTUAL residual, not by |df|.  The discrete state is
    # not exactly the local equilibrium -- frame recovery returns (kD,Te,mu)
    # from grid moments, so df_loc is small but nonzero -- and damping that
    # genuine residual is precisely the closure's job.  The invariant that
    # must hold is that the closure never exceeds gamma_res * |df_loc|, i.e.
    # it damps what is there and does not invent a source of its own.
    f = rep._f0_lab[None] + df
    kD_r, Te_r, mu_r = rep._recover_frame(f)
    kp = rep.k[None] - kD_r[:, None]
    xi = (kp.square().sum(-1) / (2 * fs_on.m_star) - mu_r[:, None]) / Te_r[:, None]
    df_loc_max = float((f - torch.special.expit(-xi)).abs().max())
    rel = float((a - b).abs().max()) / (fs_on.gamma_residual() * df_loc_max)
    assert df_loc_max < 0.05 * float(df.abs().max()), "frame recovery is off"
    assert rel < 1.0, (
        f"closure output is {rel:.3f} x gamma_res*|df_loc| -- it is producing "
        "more than a relaxation of the residual actually present"
    )


def test_residual_closure_spares_odd_harmonics() -> None:
    """Odd angular harmonics are gated to zero at leading order; the parity
    split must leave them essentially untouched."""
    fs = _make_cart_closure()
    rep = fs.representation
    kD = 0.37 * rep._dk_grid * np.array([1.0, 0.61])
    kp = rep.k - torch.as_tensor(kD, device=rep.k.device, dtype=rep.k.dtype)
    xi = (kp.square().sum(-1) / (2 * fs.m_star) - fs.mu) / fs.T
    th = torch.atan2(kp[:, 1], kp[:, 0])
    f0 = torch.special.expit(-xi)
    # a HIGH odd harmonic, above M_theta = 6, so nothing else acts on it:
    df = ((f0 * (1 - f0) / fs.T) * torch.cos(9 * th))[None]
    out = rep.apply_collision(df.clone(), lambda a, te=None: torch.zeros_like(a))
    leak = float(out.abs().max()) / (fs.gamma_residual() * float(df.abs().max()))
    # MEASURED 0.057 at dk = T/(3 vF).  This is the bilinear interpolation
    # error of the drift reflection, not a parity-logic error: without the
    # split the same mode would be damped at 1.0 x gamma_res, so the gate buys
    # ~20x.  The physical odd-m rate is ~0.01 gamma_2, so the artifact is still
    # ~20x the physics it protects -- a bicubic (16-corner, O(h^4)) gather
    # should reach ~0.5%.  Threshold here is a regression guard.
    assert leak < 8e-2, f"odd harmonic damped at {leak:.3f} of gamma_res"


def test_residual_closure_damps_even_harmonics() -> None:
    """The even counterpart of the previous test MUST be damped, at ~gamma_res
    and with the sign of a decay."""
    fs = _make_cart_closure()
    rep = fs.representation
    kD = 0.37 * rep._dk_grid * np.array([1.0, 0.61])
    kp = rep.k - torch.as_tensor(kD, device=rep.k.device, dtype=rep.k.dtype)
    xi = (kp.square().sum(-1) / (2 * fs.m_star) - fs.mu) / fs.T
    th = torch.atan2(kp[:, 1], kp[:, 0])
    f0 = torch.special.expit(-xi)
    df = ((f0 * (1 - f0) / fs.T) * torch.cos(10 * th))[None]
    out = rep.apply_collision(df.clone(), lambda a, te=None: torch.zeros_like(a))
    ov = float((out[0] * df[0]).sum() / (df[0] * df[0]).sum())
    assert ov < 0.0, "even residual is not damped"
    assert (
        0.5 < abs(ov) / fs.gamma_residual() < 1.5
    ), f"even residual damped at {abs(ov) / fs.gamma_residual():.2f} x gamma_res"


def test_cartesian_annulus_with_frame_polish() -> None:
    T = 0.05
    fs = FermiSurface(
        kF=1.0,
        vF=1.5,
        n_theta=6,
        n_r=4,
        T=T,
        xi_max=6.0,
        tau_ee=200.0,
        cartesian=dict(dk=T / 4.5, annulus_xi=8.0, local_te_rates=False),
        process_grid=get_process_grid("rk", (1, 1)),
    )
    rep = fs.representation
    assert rep._annulus_on, "annulus did not engage: test is vacuous"
    assert rep._k_polish is not None
    assert rep._k_polish.shape[0] > rep.k.shape[0]
    assert rep._k_polish.shape[0] == rep._eps_polish.shape[0]
    torch.manual_seed(0)
    df = 0.01 * torch.randn(3, rep.Nk, device=rep.k.device, dtype=rep.k.dtype)
    out = rep.apply_collision(df, fs._modal_collision)
    assert torch.isfinite(out).all()


def test_residual_closure_off_identical() -> None:
    fs_off = FermiSurface(
        kF=1.0,
        vF=1.5,
        n_theta=6,
        n_r=4,
        T=0.05,
        xi_max=6.0,
        tau_ee=200.0,
        cartesian=dict(dk=0.05 / 4.5, local_te_rates=False),
        process_grid=get_process_grid("rk", (1, 1)),
    )
    rep = fs_off.representation
    torch.manual_seed(0)
    df = 0.01 * torch.randn(3, rep.Nk, device=rep.k.device, dtype=rep.k.dtype)
    a = rep.apply_collision(df.clone(), fs_off._modal_collision)
    b = rep.apply_collision(df.clone(), fs_off._modal_collision)
    assert torch.equal(a, b), "collision apply is not deterministic"
    assert fs_off.gamma_residual() is None


@pytest.mark.parametrize("n_r", [2, 4])
@pytest.mark.parametrize("r_c", [np.inf, 2.0])
def test_collision_conserves_charge_radial(n_r: int, r_c: float) -> None:
    fs = _make(n_theta=8, n_r=n_r, tau_p=2.0, tau_ee=3.0, r_c=r_c)
    rng = torch.Generator(device=rc.device).manual_seed(2)
    rho = torch.randn(7, fs.v.shape[0], dtype=torch.float64, generator=rng)
    rdot = fs.rho_dot(rho, 0.0, 0)
    mdot = fs.measure_observables(rdot, 0.0)[0, 0]
    mass = fs.measure_observables(rho, 0.0)[0, 0]
    assert float((mdot / mass.abs().clamp(min=1e-30)).abs().max()) < 1e-11


@pytest.mark.parametrize("phi_deg", [23.7, 45.0, 90.0])
@pytest.mark.parametrize("s", [0.0, 0.7])
def test_reflector_conserves_charge_radial(phi_deg: float, s: float) -> None:
    """Net (w_r-weighted) normal mass flux at the wall is zero at Nr>1.  Only the
    n=0 radial projection carries mass; the per-radial (D, T) solve conserves it."""
    n_r = 4
    fs = _make(n_theta=8, n_r=n_r, specularity=s)
    phi = np.deg2rad(phi_deg)
    n = torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=torch.float64)
    refl = _DeltaKReflector(fs, n, specularity=s)
    rng = torch.Generator(device=rc.device).manual_seed(0)
    u_in = torch.randn(1, 1, fs.v.shape[0], dtype=torch.float64, generator=rng)
    u_out = refl(u_in)
    ui = u_in.reshape(n_r, fs.n_theta)
    uo = u_out.reshape(n_r, fs.n_theta)
    w_r = fs.radial.w_xi
    theta = fs.angular.theta
    vdn = fs.vF * (n[0, 0] * torch.cos(theta) + n[0, 1] * torch.sin(theta))
    F_out = (w_r[:, None] * vdn.clamp(min=0)[None, :] * ui).sum()
    F_in = (w_r[:, None] * vdn.clamp(max=0)[None, :] * uo).sum()
    assert float((F_out + F_in).abs()) / max(float(F_out.abs()), 1e-30) < 1e-11


@pytest.mark.parametrize("phi_deg", [23.7, 45.0, 90.0])
@pytest.mark.parametrize("specularity", [0.0, 0.7])
def test_reflector_tangential_momentum_radial(
    phi_deg: float, specularity: float
) -> None:
    """Specular reflector should conserve tangetial momentum,
    while diffuse reflector should scale it by (1 - specularity)."""
    n_r = 4
    fs = _make(n_theta=8, n_r=n_r, specularity=specularity)
    phi = np.deg2rad(phi_deg)
    n = torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=torch.float64)
    refl = _DeltaKReflector(fs, n, specularity=specularity)
    rng = torch.Generator(device=rc.device).manual_seed(0)
    u_in = torch.randn(1, 1, fs.v.shape[0], dtype=torch.float64, generator=rng)
    u_out = refl(u_in)
    ui = u_in.reshape(n_r, fs.n_theta)
    uo = u_out.reshape(n_r, fs.n_theta)
    w_r = fs.radial.w_xi
    theta = fs.angular.theta
    vdn = fs.vF * (n[0, 0] * torch.cos(theta) + n[0, 1] * torch.sin(theta))
    vtg = fs.vF * (-n[0, 1] * torch.cos(theta) + n[0, 0] * torch.sin(theta))
    F_out = (w_r[:, None] * (vdn.clamp(min=0) * vtg)[None, :] * ui).sum()
    F_in = (w_r[:, None] * (vdn.clamp(max=0) * vtg)[None, :] * uo).sum()
    expected = (1.0 - specularity) * F_out
    assert (
        float((F_out + F_in - expected).abs()) / max(float(F_out.abs()), 1e-30) < 1e-11
    )


@pytest.mark.parametrize("n_r", [2, 4])
def test_tau_ee_damps_energy(n_r: int) -> None:
    """The empirical `tau_ee` model cannot conserve energy."""
    tau_ee = 2.0
    fs = _make(n_theta=8, n_r=n_r, tau_p=np.inf, tau_ee=tau_ee, r_c=np.inf)
    w_r, xi = fs.radial.w_xi, fs.radial.xi
    E_obs = ((w_r * xi)[:, None].expand(-1, fs.n_theta) / fs.n_theta).flatten()
    rng = torch.Generator(device=rc.device).manual_seed(3)
    rho = torch.randn(9, fs.v.shape[0], dtype=torch.float64, generator=rng)
    rdot = fs.rho_dot(rho, 0.0, 0)
    e_rate = (E_obs[None, :] * rdot).sum(-1)
    e_val = (E_obs[None, :] * rho).sum(-1)
    # exact exponential decay of the energy moment at the placeholder's own rate
    ratio = e_rate / e_val.abs().clamp(min=1e-30).copysign(e_val)
    assert float((ratio + 1.0 / tau_ee).abs().max()) < 1e-11

    # ...and the rate table it comes from, read directly.  This is the whole
    # content of the operator, so assert every block of it:
    rates = fs.rates_modal.reshape(n_r, fs.n_theta)
    m_max = fs.angular.m_max
    assert float(rates[0, 0]) == 0.0  # charge: protected
    assert float(rates[0, 1]) == 0.0  # momentum: protected
    assert float(rates[0, m_max + 1]) == 0.0  # (tau_p = inf)
    assert float((rates[1:, 0] - 1.0 / tau_ee).abs().max()) < 1e-15  # energy is not
    assert float((rates[1:, 1] - 1.0 / tau_ee).abs().max()) < 1e-15
    assert float((rates[1:, m_max + 1] - 1.0 / tau_ee).abs().max()) < 1e-15


@pytest.mark.parametrize("n_r", [1, 4])
def test_tau_ee_conserves_momentum(n_r: int) -> None:
    """The empirical `tau_ee` model must conserve momentum at zero'th radial mode."""
    tau_ee = 2.0
    fs = _make(n_theta=8, n_r=n_r, tau_p=np.inf, tau_ee=tau_ee, r_c=np.inf)
    rates = fs.rates_modal.reshape(n_r, fs.n_theta)
    m_max = fs.angular.m_max
    assert float(rates[0, 1]) == 0.0 and float(rates[0, m_max + 1]) == 0.0
    w_r = fs.radial.w_xi
    Jx = (w_r[:, None] * torch.cos(fs.angular.theta)[None, :]).reshape(-1)
    rng = torch.Generator(device=rc.device).manual_seed(5)
    rho = torch.randn(6, fs.v.shape[0], dtype=torch.float64, generator=rng)
    rdot = fs.rho_dot(rho, 0.0, 0)
    j_rate = (Jx[None, :] * rdot).sum(-1)
    j_val = (Jx[None, :] * rho).sum(-1)
    ratio = float((j_rate / j_val.abs().clamp(min=1e-30)).abs().max())
    assert ratio < 1e-11, f"momentum decaying at {ratio:.3e}"


@pytest.mark.parametrize("n_r", [1, 4])
def test_tau_ee_momentum_leak_bounded(n_r: int) -> None:
    """Test higher radial mode leakage of momentum in `tau_ee` model is bounded."""
    tau_ee = 2.0
    kF, vF, T = 7.5e-3, 0.11194, 1.3301e-5  # mixer material
    t = T / (0.5 * kF * vF)
    fs = FermiSurface(
        kF=kF,
        vF=vF,
        n_theta=8,
        n_r=n_r,
        T=T,
        xi_max=6.0,
        tau_p=np.inf,
        tau_ee=tau_ee,
        r_c=np.inf,
        specularity=1.0,
        process_grid=get_process_grid("rk", (1, 1)),
    )
    w_r, xi = fs.radial.w_xi, fs.radial.xi
    arg = 1.0 + t * xi
    assert float(arg.min()) > 0.0, "band bottom inside the xi window"
    J = (
        w_r[:, None] * torch.sqrt(arg)[:, None] * torch.cos(fs.angular.theta)[None, :]
    ).reshape(-1)
    rho = J[None, :].clone()  # denominator = <J,J>, never small
    rate = -float((J[None, :] * fs.rho_dot(rho, 0.0, 0)).sum()) / float((J * J).sum())
    assert rate >= -1e-15, f"momentum GROWING at {rate:.3e}"
    assert rate * tau_ee < 1e-3, f"leak {rate * tau_ee:.3e} exceeds the measured 6.1e-4"
    if n_r == 1:
        assert rate * tau_ee < 1e-20, "Fermi circle must be exact"


@pytest.mark.parametrize("deg", [0.0, 45.0, 90.0])
def test_reflector_aligned_walls_exact(deg: float) -> None:
    kF, vF, T = 7.5e-3, 0.11194, 1.3301e-5
    fs = FermiSurface(
        kF=kF,
        vF=vF,
        n_theta=8,
        n_r=2,
        T=T,
        xi_max=6.0,
        tau_p=np.inf,
        specularity=1.0,
        cartesian=dict(
            annulus_xi=0.0,
            te_fac_max=6.0,
            kD_max=1.2e-3,
            dmu_max=1.2e-4,
            k_max=0.0132557160008,
            n_k=28,
        ),
        process_grid=get_process_grid("rk", (1, 1)),
    )
    rep = fs.representation
    k = rep.k
    m, mu = float(rep.m_star), float(rep.mu)
    f0 = rep._f0_lab
    kD = torch.tensor([0.05 * kF, 0.0], dtype=k.dtype, device=k.device)
    u_in = torch.special.expit(-(((k - kD) ** 2).sum(-1) / (2 * m) - mu) / T) - f0
    phi = np.deg2rad(deg)
    n = torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=k.dtype, device=k.device)
    out = fs.get_reflector(n)(u_in[None, :])[0]
    kn = (k * n[0]).sum(-1, keepdim=True)
    kstar = k - 2.0 * kn * n[0][None, :]
    exact = torch.special.expit(-(((kstar - kD) ** 2).sum(-1) / (2 * m) - mu) / T) - f0
    scale = exact.abs().max().clamp(min=1e-300)
    assert float((out - exact).abs().max() / scale) < 1e-12
    assert float((f0 + out).min()) >= -1e-14, "occupancy went negative"


def test_reflector_tilted_walls_bounded() -> None:
    """Check that reflections that require interpolation keep f near [0, 1]."""
    kF, vF, T = 7.5e-3, 0.11194, 1.3301e-5
    fs = FermiSurface(
        kF=kF,
        vF=vF,
        n_theta=32,
        n_r=6,
        T=T,
        xi_max=6.0,
        tau_p=np.inf,
        specularity=1.0,
        cartesian=dict(
            annulus_xi=0.0,
            te_fac_max=6.0,
            kD_max=1.2e-3,
            dmu_max=1.2e-4,
            k_max=0.0132557160008,
            n_k=224,
        ),
        process_grid=get_process_grid("rk", (1, 1)),
    )
    rep = fs.representation
    k = rep.k
    m, mu = float(rep.m_star), float(rep.mu)
    f0 = rep._f0_lab
    kD = torch.tensor([0.05 * kF, 0.0], dtype=k.dtype, device=k.device)
    u_in = torch.special.expit(-(((k - kD) ** 2).sum(-1) / (2 * m) - mu) / T) - f0
    phi = np.deg2rad(17.0)
    n = torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=k.dtype, device=k.device)
    out = fs.get_reflector(n)(u_in[None, None, :])[0, 0]
    kn = (k * n[0]).sum(-1, keepdim=True)
    kstar = k - 2.0 * kn * n[0][None, :]
    exact = torch.special.expit(-(((kstar - kD) ** 2).sum(-1) / (2 * m) - mu) / T) - f0
    xi = (k.square().sum(-1) / (2 * m) - mu) / T
    shell = xi.abs() < 6
    rel = float(
        (out - exact)[shell].abs().max() / exact[shell].abs().max().clamp(min=1e-300)
    )
    assert rel < 2e-2, f"tilted-wall shell error {rel:.3e} back in the O(10%) regime"


@pytest.mark.parametrize("deg", [23.7, 45.0, 90.0])
def test_specular_rotation_exact(deg: float) -> None:
    fs = FermiSurface(
        kF=1.0,
        vF=1.5,
        n_theta=32,
        n_r=1,
        T=1.0,
        tau_p=np.inf,
        specularity=1.0,
        process_grid=get_process_grid("rk", (1, 1)),
    )
    rep = fs.representation
    th = fs.angular.theta.clone()
    g = torch.Generator(device=rc.device).manual_seed(1)
    m_max = fs.angular.m_max
    c = torch.randn(m_max + 1, generator=g, device=rc.device, dtype=torch.float64)
    s = torch.randn(m_max, generator=g, device=rc.device, dtype=torch.float64)
    m = torch.arange(m_max + 1, device=rc.device)

    def series(x):
        m_x = x[:, None] * m
        return m_x.cos() @ c + m_x[..., :m_max].sin() @ s

    u = series(th)
    phi = np.deg2rad(deg)
    n = torch.tensor(
        [[np.cos(phi), np.sin(phi)]], dtype=torch.float64, device=th.device
    )
    os.environ["QIMPY_REFL_BUDGET_GB"] = "0.0"  # avoid DenseReflector wrap
    R = fs.get_reflector(n)
    rot = rep.from_modes(R._specular_modal(rep.to_modes(u[None, None, :])))[0, 0]
    exact = series(2 * (phi + np.pi / 2) - th)
    assert float((rot - exact).abs().max() / u.abs().max()) < 1e-12


def test_cartesian_wall_bounded() -> None:
    pg = get_process_grid("rk", (1, 1))
    material = FermiSurface(
        process_grid=pg,
        kF=7.5e-3,
        vF=0.11194,
        n_theta=32,
        n_r=6,
        T=1.3301e-5,
        xi_max=6.0,
        tau_p=np.inf,
        tau_ee=np.inf,
        specularity=1.0,
        cartesian=dict(annulus_xi=0.0, te_fac_max=6.0),
    )
    rep = material.representation
    f0 = rep._f0_lab
    th = torch.linspace(0.0, 2 * np.pi, 17, device=rc.device)[:-1]
    n = torch.stack([th.cos(), th.sin()], -1)
    kD = torch.tensor([6.0e-5, 0.0], device=rc.device)
    eps = ((rep.k - kD) ** 2).sum(-1) / (2 * rep.m_star)
    for te_fac in (1.0, 5.6):
        f_tr = torch.special.expit(-(eps - rep.mu) / (te_fac * rep.T))
        u = (f_tr - f0)[None].repeat(n.shape[0], 1)
        for s in (1.0, 0.5):
            rep.fs.specularity = s
            refl = rep.get_reflector(n)
            f = f0[None] + refl(u[None])[0]
            assert float(f.min()) > -1e-14, (te_fac, s, float(f.min()))
            if s == 1.0:
                assert float(f.max()) < 1.0 + 1e-14, (te_fac, s, float(f.max()))
