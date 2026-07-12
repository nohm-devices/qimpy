"""FermiSurface: unified Fermi-surface / Fermi-circle material.

Storage: delta-k collocation in (k_r, theta_q), shape (Np, K, Nr*N_theta).
Modal transforms: tensor product of an angular Fourier basis (over theta) and a
radial polynomial basis (over xi = (E-mu)/T) orthonormal under the equilibrium
fluctuation weight w(xi) = (1/4T) sech^2(xi/2). The radial basis collapses to
the identity at Nr=1, recovering the Fermi-circle limit of a single Fermi-
surface state with angular Fourier modes.

Per-operator basis is chosen for each operator's natural form:
    - real-space advection      : delta-k, scalar per-collocation upwind
    - cyclotron / collision     : modal, diagonal in (l, n)
    - specular / diffuse walls  : modal (handled inside the reflector object)
    - contacts                  : constructed in modal, transformed to delta-k once
    - observables {n, jx, jy}   : delta-k weighted sums
"""
from __future__ import annotations
from typing import Callable, Optional, Union
import numpy as np
import torch

from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.profiler import stopwatch
from qimpy.io import CheckpointPath, CheckpointContext, InvalidInputException
from .scattering import EEScattering
from .._material import Material


# ----------------------------------------------------------------------------
# Angular basis: real Fourier on [0, 2*pi), nodal storage at theta_q = 2*pi*q/Nq
# ----------------------------------------------------------------------------
class AngularBasis:
    """Real Fourier transforms with nodal (delta-k) storage.

    Mode ordering: ``(a_0, a_1, b_1, a_2, b_2, ..., a_M, b_M)`` for ``2M+1`` modes.
    Nodes are the *midpoint* grid ``theta_q = 2*pi*(q+1/2)/Nq``; the transforms
    are exact inverses for any ``n_quad = Nq >= 2M+1``.  The midpoint grid is
    invariant under both ``theta->-theta`` and (for even ``Nq``) ``theta->pi-theta``
    -- the ``v_y->-v_y`` and ``v_x->-v_x`` wall reflections -- so callers that need
    left-right contact symmetry pass an even ``Nq`` (see ``FermiSurface``).
    """

    def __init__(self, M: int, n_quad: int | None = None,
                 dtype: torch.dtype = torch.float64,
                 device: torch.device | None = None) -> None:
        Nq = (2 * M + 1) if n_quad is None else n_quad
        if Nq < 2 * M + 1:
            raise ValueError(f"AngularBasis n_quad={Nq} < 2M+1={2*M+1}")
        self.M = M
        self.dim = 2 * M + 1
        self.N_theta = Nq
        self.wphi = 2.0 * np.pi / Nq          # uniform angular integration weight
        # Midpoint (half-offset) nodes.  Unlike the endpoint grid 2*pi*q/Nq, this
        # set is symmetric under theta->-theta and (even Nq) theta->pi-theta, i.e.
        # the v_y->-v_y and v_x->-v_x reflections; the endpoint grid breaks
        # v_x->-v_x for odd Nq, destroying left-right contact symmetry in the
        # collisionless limit.  Discrete Fourier orthogonality holds for Nq>=2M+1.
        theta = 2.0 * np.pi * (np.arange(Nq) + 0.5) / Nq
        # modes -> nodes:  T_from_modes[q, c] s.t. f(theta_q) = sum_c T_fm[q,c] a_c
        Tfm = np.zeros((Nq, self.dim))
        Tfm[:, 0] = 1.0
        for m in range(1, M + 1):
            Tfm[:, 2 * m - 1] = np.cos(m * theta)
            Tfm[:, 2 * m]     = np.sin(m * theta)
        # nodes -> modes (discrete Fourier coefficients)
        Ttm = np.zeros((self.dim, Nq))
        Ttm[0, :] = 1.0 / Nq
        for m in range(1, M + 1):
            Ttm[2 * m - 1, :] = (2.0 / Nq) * np.cos(m * theta)
            Ttm[2 * m, :]     = (2.0 / Nq) * np.sin(m * theta)
        # cyclotron generator G: block-skew per harmonic; (G a)_m = m*(b_m,-a_m)
        G = np.zeros((self.dim, self.dim))
        for m in range(1, M + 1):
            G[2 * m - 1, 2 * m] = -m
            G[2 * m,     2 * m - 1] = +m
        dev = device or rc.device
        self.theta        = torch.as_tensor(theta, dtype=dtype, device=dev)
        self.T_from_modes = torch.as_tensor(Tfm,   dtype=dtype, device=dev)
        self.T_to_modes   = torch.as_tensor(Ttm,   dtype=dtype, device=dev)
        self.G            = torch.as_tensor(G,     dtype=dtype, device=dev)


# ----------------------------------------------------------------------------
# Radial basis: polynomials in xi orthonormal under  w(xi) = (1/4T) sech^2(xi/2)
# ----------------------------------------------------------------------------
class RadialBasis:
    """Polynomial transforms orthonormal under the equilibrium-fluctuation
    weight ``w(xi) = (1/4T) sech^2(xi/2)``.

    For ``Nr == 1`` the basis collapses to the identity at a single point
    ``xi = 0`` (Fermi surface); this is the regime where the material is the
    pure Fermi-circle limit -- one Fermi-surface state, angular Fourier modes
    only.  For ``Nr > 1`` we use ``Nr`` Gauss-Legendre points on
    ``[-xi_max, xi_max]`` as collocation, build the polynomial powers and
    orthonormalize them by Cholesky of the discrete mass matrix under the
    combined ``(Gauss-Legendre weight) * (sech^2 weight)``; this guarantees the
    discrete transforms ``T_to_modes @ T_from_modes = I`` exactly.
    """

    def __init__(self, Nr: int, T_temp: float = 1.0, xi_max: float = 6.0,
                 dtype: torch.dtype = torch.float64,
                 device: torch.device | None = None) -> None:
        self.Nr     = Nr
        self.T_temp = T_temp
        self.xi_max = xi_max
        dev = device or rc.device
        if Nr == 1:
            # Trivial: single point at the Fermi surface, identity transforms.
            self.xi           = torch.zeros(1, dtype=dtype, device=dev)
            self.quad_w       = torch.ones(1,  dtype=dtype, device=dev)
            self.flat_w       = torch.ones(1,  dtype=dtype, device=dev)  # flat phase-space measure
            self.T_from_modes = torch.ones((1, 1), dtype=dtype, device=dev)
            self.T_to_modes   = torch.ones((1, 1), dtype=dtype, device=dev)
            return
        # Gauss-Legendre nodes on [-1, 1] scaled to [-xi_max, xi_max].
        x_std, w_std = np.polynomial.legendre.leggauss(Nr)
        xi  = xi_max * x_std
        w_x = xi_max * w_std                                 # Jacobian
        w_eq = (1.0 / (4.0 * T_temp)) / np.cosh(0.5 * xi) ** 2
        w_q  = w_x * w_eq                                    # discrete measure
        # Orthonormalize the polynomial powers under the discrete weight <,>_w
        # by QR of the WEIGHTED Vandermonde, in the RESCALED variable
        # u = xi/xi_max (so the monomials u^p stay O(1)).  This is far better
        # conditioned than Cholesky of the mass matrix G = Vw^T Vw: QR works on
        # Vw directly, whose condition number is the square ROOT of G's, so the
        # construction stays accurate to much higher Nr (Cholesky(G) already
        # fails its orthonormality check at Nr >= 14).  The result is the SAME
        # orthonormal polynomial basis (unique up to sign); the sign is fixed to
        # a positive R-diagonal, matching the previous Cholesky convention, so
        # the basis (and everything built on it) is unchanged where both work.
        V  = np.vander(xi / xi_max, Nr, increasing=True)     # monomials in u
        Vw = V * np.sqrt(w_q)[:, None]                       # Vw^T Vw = G
        _Q, R = np.linalg.qr(Vw)
        sgn = np.sign(np.diag(R))
        sgn[sgn == 0] = 1.0
        R = sgn[:, None] * R                                 # positive diagonal
        Tfm = V @ np.linalg.solve(R, np.eye(Nr))             # nodes <- modes
        Ttm = Tfm.T * w_q                                    # modes <- nodes
        # Sanity: Ttm @ Tfm should be identity (orthonormality under <,>_w).
        eye_check = Ttm @ Tfm
        err = float(np.max(np.abs(eye_check - np.eye(Nr))))
        if err > 1e-10:
            raise RuntimeError(
                f"RadialBasis: orthonormality check failed (max |Ttm@Tfm - I| = {err:.2e})"
            )
        self.xi           = torch.as_tensor(xi,  dtype=dtype, device=dev)
        self.quad_w       = torch.as_tensor(w_q, dtype=dtype, device=dev)
        self.flat_w       = torch.as_tensor(w_x, dtype=dtype, device=dev)  # flat GL phase-space measure
        self.T_from_modes = torch.as_tensor(Tfm, dtype=dtype, device=dev)
        self.T_to_modes   = torch.as_tensor(Ttm, dtype=dtype, device=dev)


# ----------------------------------------------------------------------------
# FermiSurface material
# ----------------------------------------------------------------------------
class FermiSurface(Material):
    """Unified Fermi-surface material; ``Nr=1`` recovers the Fermi-circle limit.

    Parameters
    ----------
    kF, vF
        Fermi wave-vector and Fermi velocity (a.u.).
    M_theta
        Highest angular harmonic retained (storage carries ``2*M_theta+1`` per
        radial point).
    Nr
        Radial mode count (defaults to 1 = Fermi-circle limit).
    T
        Temperature in energy units (a.u.).  Used only when ``Nr > 1``.
    xi_max
        Radial truncation in units of ``T`` (defaults to 6: tails of sech^2 are
        ``< 1e-5``).
    tau_p, tau_ee, r_c, specularity
        Phenomenological momentum-relaxation time, electron-electron-collision
        time, cyclotron radius (``inf`` for no field), and wall specularity
        ``s in [0, 1]`` (``s=1`` pure specular, ``s=0`` fully diffuse).
    ee_scattering
        Microscopic e-e collision operator (`scattering.EEScattering`);
        mutually exclusive with the phenomenological ``tau_ee``.
    """

    kF: float; vF: float; M_theta: int; Nr: int
    T_temp: float; xi_max: float
    tau_inv_p: float; tau_inv_ee: float
    r_c: float; specularity: float
    k_speed: float
    angular: AngularBasis
    radial:  RadialBasis

    def __init__(
        self, *, kF: float, vF: float, M_theta: int,
        Nr: int = 1, T: float = 1.0, xi_max: float = 6.0,
        tau_p: float = np.inf, tau_ee: float = np.inf,
        r_c: float = np.inf, specularity: float = 1.0,
        ee_scattering: Optional[Union[EEScattering, dict]] = None,
        moving_frame: bool = False,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        super().__init__()
        self.moving_frame = bool(moving_frame)
        self.kF, self.vF = kF, vF
        self.M_theta, self.Nr = M_theta, Nr
        self.T_temp, self.xi_max = T, xi_max
        self.r_c = r_c
        self.tau_inv_p  = 1.0 / tau_p
        self.tau_inv_ee = 1.0 / tau_ee
        self.specularity = specularity
        # The nodal state is the FULL distribution f in [0,1]: initialized to the
        # equilibrium Fermi-Dirac f0, with contacts/diffuse walls re-emitting a
        # genuine Fermi-Dirac occupation.
        # Fermi energy for the band group-velocity speed factor (parabolic band:
        # E_F = kF^2/(2 m*) = kF * vF / 2, since vF = kF/m*).
        self.E_F = 0.5 * kF * vF
        # Moving drift-frame constants (atomic units, hbar=1; parabolic 2D band).
        # m* = kF/vF; spin gs=2; 2D DOS g = gs m*/(2 pi); k-integral norm gs/(2pi)^2.
        self.hbar = 1.0
        self.mstar = kF / vF
        self.gs = 2.0
        self.g2d = self.gs * self.mstar / (2.0 * np.pi)
        self.cnorm = self.gs / (2.0 * np.pi) ** 2
        # Even angular-node count (rounded up to a multiple of 4, >= 2M+1): with
        # the midpoint quadrature this is symmetric under both v_x->-v_x and
        # v_y->-v_y and places no node tangent to an axis-aligned wall (avoids a
        # 0/0 in the specular reflector).  Keeps source/drain (left-right)
        # symmetry exact even in the ballistic limit; an odd 2M+1 grid breaks it.
        # The retained mode count (2M+1) is unchanged.
        N_theta = -(-(2 * M_theta + 1) // 4) * 4
        N_k = Nr * N_theta
        self.initialize(wk=1.0, nk=N_k, n_bands=1, n_dim=2,
                        process_grid=process_grid)
        if self.comm.size > 1:
            raise InvalidInputException(
                "FermiSurface couples k-channels at the boundary; the k "
                "process-grid dimension must be 1 (parallelize over space)."
            )
        dtype = self.v.dtype
        self.angular = AngularBasis(M_theta, n_quad=N_theta, dtype=dtype)
        self.radial  = RadialBasis(Nr, T_temp=T, xi_max=xi_max, dtype=dtype)
        # Per-collocation transport velocity = band GROUP velocity.  The speed
        # depends on the energy (radial) node through the parabolic-band factor
        #     |v(xi)| = vF * sqrt(1 + T*xi / E_F),   E_F = kF*vF/2,
        # while the direction is the angular node k_hat = (cos, sin).  At Nr=1
        # (xi=0) this collapses to the flat |v|=vF Fermi-circle limit.  The clamp
        # guards the (unphysical) case xi <= -E_F/T where a node would sit at or
        # below the band bottom (|v|->0), keeping the sqrt real.
        theta_q = self.angular.theta
        v_dir = torch.stack(
            [torch.cos(theta_q), torch.sin(theta_q)], dim=-1
        )                                                       # (N_theta, 2)
        xi_r = self.radial.xi                                   # (Nr,)
        speed_r = vF * torch.sqrt(torch.clamp(
            1.0 + (self.T_temp * xi_r) / self.E_F, min=0.0))    # (Nr,)
        self.v_speed = speed_r                                  # band |v| per node
        self.v = (speed_r[:, None, None] * v_dir[None, :, :]
                  ).reshape(N_k, 2)                             # (Nr*N_theta, 2)
        # Scalar advection -> no coupling object exposed to the DG layer
        self.coupling = None
        self.k_speed = (vF / r_c) if np.isfinite(r_c) else 0.0
        # Full-f initial / reference state: the equilibrium Fermi-Dirac occupation
        # f0(xi_r) = 1/(exp(xi_r)+1), isotropic in angle (same for every theta_q).
        # The FV geometry reads material.rho0 to seed the nodal state.
        f0_r = 1.0 / (torch.exp(xi_r) + 1.0)                   # (Nr,)
        self.rho0 = (f0_r[:, None].expand(Nr, N_theta)
                     ).reshape(-1).clone()                     # (Nr*N_theta,)
        # Per-mode collision rates in flattened (radial n, angular m) ordering.
        # Angular: m=0 conserved (rate 0); m=1 decays through tau_p only;
        # m>=2 decays through tau_p + tau_ee (both impurity and viscous channels).
        # Radial: n=0 is the equilibrium-shape mode (no extra decay); n>=1 are
        # higher energy moments that we damp with 1/tau_ee as a placeholder
        # until the microscopic-L hook is added.
        ang = np.zeros(self.angular.dim)
        for m in range(1, M_theta + 1):
            ang[2 * m - 1] = ang[2 * m] = (
                self.tau_inv_p if m == 1 else (self.tau_inv_p + self.tau_inv_ee)
            )
        rad = np.zeros(Nr)
        rad[1:] = self.tau_inv_ee
        rates = ang[None, :] + rad[:, None]                  # (Nr, dim_theta)
        # Particle conservation at (n=0, m=0): override to 0.
        rates[0, 0] = 0.0
        self.rates_modal = torch.as_tensor(
            rates.reshape(-1), dtype=dtype, device=rc.device
        )

        # Microscopic e-e collisions (replaces the tau_ee placeholder):
        if (ee_scattering is not None) or checkpoint_in.member("ee_scattering"):
            if np.isfinite(tau_ee):
                raise InvalidInputException(
                    "Specify either the phenomenological tau_ee or the"
                    " microscopic ee_scattering collision operator, not both"
                )
            self.add_child(
                "ee_scattering", EEScattering, ee_scattering, checkpoint_in,
                fermi_surface=self,
            )

    # ---- transforms (tensor product of radial and angular pieces) ----

    def to_modes(self, f: torch.Tensor) -> torch.Tensor:
        """Nodal ``(..., Nr*N_theta)`` -> modal ``(..., Nr*(2M_theta+1))``."""
        Ntheta = self.angular.N_theta
        Nr = self.Nr
        shape_in = f.shape
        f4 = f.reshape(*shape_in[:-1], Nr, Ntheta)
        # angular: (..., r, q) -> (..., r, c_theta)
        a_t = torch.einsum("cq,...rq->...rc", self.angular.T_to_modes, f4)
        # radial: (..., r, c_theta) -> (..., n_r, c_theta)
        a   = torch.einsum("nr,...rc->...nc", self.radial.T_to_modes, a_t)
        return a.reshape(*shape_in[:-1], Nr * self.angular.dim)

    def from_modes(self, a: torch.Tensor) -> torch.Tensor:
        """Modal ``(..., Nr*(2M_theta+1))`` -> nodal ``(..., Nr*N_theta)``."""
        dim_theta = self.angular.dim
        Nr = self.Nr
        shape_in = a.shape
        a4 = a.reshape(*shape_in[:-1], Nr, dim_theta)
        # radial: (..., n_r, c_theta) -> (..., r, c_theta)
        f_r = torch.einsum("rn,...nc->...rc", self.radial.T_from_modes, a4)
        # angular: (..., r, c_theta) -> (..., r, q)
        f   = torch.einsum("qc,...rc->...rq", self.angular.T_from_modes, f_r)
        return f.reshape(*shape_in[:-1], Nr * self.angular.N_theta)

    # ---- the rest, stubbed for steps 2-7 ----

    @property
    def transport_velocity(self) -> torch.Tensor:
        return self.v

    @stopwatch
    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        """Cyclotron + collision in modal space; identity in delta-k storage.

        Transforms rho (delta-k) -> a (modal), applies diagonal collision rates
        and (if r_c is finite) the exact cyclotron generator G acting on the
        angular block within each radial mode, then transforms back.
        """
        has_ee = hasattr(self, "ee_scattering")
        if (
            self.rates_modal.abs().sum() == 0
            and self.k_speed == 0.0
            and not has_ee
        ):
            return torch.zeros_like(rho)                       # ballistic, no field
        a = self.to_modes(rho)                                 # (..., Nr*dim_theta)
        a_dot = -self.rates_modal * a
        if has_ee:
            a_dot = a_dot + self.ee_scattering.a_dot(a)
        if self.k_speed:
            Nr, dim_t = self.Nr, self.angular.dim
            a4 = a.reshape(*a.shape[:-1], Nr, dim_t)
            Ga4 = torch.einsum("dc,...nc->...nd", self.angular.G, a4)
            a_dot = a_dot + self.k_speed * Ga4.reshape(*a.shape)
        return self.from_modes(a_dot)

    def get_observable_names(self) -> list[str]:
        return ["n", "jx", "jy"]

    @stopwatch
    def get_observables(self, t: float) -> torch.Tensor:
        """Per-channel coefficients for [n, jx, jy] in delta-k storage.

        n[r,q]  = w_r / N_theta
        jx[r,q] = w_r * |v_r| * cos(theta_q) / N_theta
        jy[r,q] = w_r * |v_r| * sin(theta_q) / N_theta

        The current uses the band GROUP speed |v_r| = vF*sqrt(1 + T*xi_r/E_F)
        per energy node (self.v_speed), consistent with the transport velocity;
        at Nr=1 |v_r| = vF and these are the standard Fermi-circle integrals.
        For Nr>1 the radial weights w_r are the equilibrium-fluctuation-weighted
        Gauss-Legendre weights from RadialBasis.
        """
        N_theta = self.angular.N_theta
        theta = self.angular.theta
        # Normalize the radial weights by sqrt(sum quad_w) so the observable is
        # the n=0 radial-mode amplitude a_0 for ANY Nr (Nr-consistent).  Without
        # this, an n=0-driven state reports sum(quad_w * psi_0) = sqrt(sum quad_w)
        # times a_0 -- which is 1 for Nr=1 but ~1/sqrt(T) for Nr>1, the spurious
        # ~248x mismatch between Nr=1 and Nr=4.  No-op at Nr=1 (sum quad_w = 1).
        w_r = self.radial.quad_w / torch.sqrt(self.radial.quad_w.sum())  # (Nr,)
        cos_q = torch.cos(theta) / N_theta                    # (N_theta,)
        sin_q = torch.sin(theta) / N_theta
        one_q = torch.full_like(cos_q, 1.0 / N_theta)
        # (Nr, N_theta) -> flatten to (Nr * N_theta,)
        n_rq  = (w_r[:, None] * one_q[None, :]).reshape(-1)
        jx_rq = (w_r[:, None] * self.v_speed[:, None] * cos_q[None, :]).reshape(-1)
        jy_rq = (w_r[:, None] * self.v_speed[:, None] * sin_q[None, :]).reshape(-1)
        return torch.stack([n_rq, jx_rq, jy_rq], dim=0)

    # ================================================================
    # Moving drift-frame kit (Stage 1): conserved-density closure, shell
    # moments, lab fluxes, and the moment-free projection.  Everything is
    # batched over a leading cell axis (...,) and lives in momentum space
    # only (no geometry).  It uses the FLAT phase-space measure radial.flat_w
    # (NOT the sech^2 quad_w, which stays inside the collision operator) --
    # the two-metric rule.  f0 = self.rho0 = sigma(-xi) is the frame reference.
    # ================================================================
    def _kbar(self, mu, Te):
        """|k_bar|(xi_r) per (cell, radial node): sqrt(2 m*(mu+Te xi'))/hbar. (...,Nr).
        eps is floored > 0 so k̄ never reaches the polar-origin singularity (1/k̄ in the
        grid velocity φ̇): a node crosses the band bottom (eps<0) only when the gas heats
        out of the degenerate regime (mu/Te < |xi'_inner|), and such nodes are deep-filled
        core (f≈1, deviation≈0) -> the floor is physically inert but keeps φ̇ finite."""
        eps = mu[..., None] + Te[..., None] * self.radial.xi
        return torch.sqrt(2.0 * self.mstar * eps.clamp_min(1e-6 * self.E_F)) / self.hbar

    def n_FD(self, mu, Te):
        """2D drifted-heated-FD density  g Te ln(1+e^{mu/Te})."""
        return self.g2d * Te * torch.logaddexp(torch.zeros_like(mu), mu / Te)

    def Eth_FD(self, mu, Te):
        """Frame thermal energy density  g Te^2 (-Li2(-e^{mu/Te})), valid at ANY
        mu/Te (degenerate AND heated).  Uses the dilog inversion so both branches
        expand in |t|=e^{-|x|} <= 1: for x>=0 the Sommerfeld form 1/2 x^2 + pi^2/6
        minus the convergent tail; for x<0 the direct alternating series.  This is
        essential once strong collisionless shear heating drives mu/Te toward 0."""
        x = mu / Te
        tpos = torch.exp(-torch.clamp(x, min=0.0))       # e^{-x} for x>=0
        tneg = torch.exp(torch.clamp(x, max=0.0))        # e^{x}  for x<0

        def _neg_li2_neg(t):                             # -Li2(-t) = sum (-1)^{k+1} t^k/k^2
            s = torch.zeros_like(t); tk = torch.ones_like(t)
            for k in range(1, 97):                       # 96 terms: machine for |mu/Te|>~0.1
                tk = tk * t
                s = s + ((1.0 if k % 2 else -1.0) / (k * k)) * tk
            return s
        Fpos = 0.5 * x * x + (np.pi ** 2 / 6.0) - _neg_li2_neg(tpos)
        Fneg = _neg_li2_neg(tneg)
        return self.g2d * Te * Te * torch.where(x >= 0.0, Fpos, Fneg)

    def mu_of_nT(self, n, Te):
        """Invert n = g Te ln(1+e^{mu/Te}): mu = Te (y + ln(1 - e^{-y})), y=n/(g Te).
        y floored so ln(1-e^{-y}) stays finite (mu->-inf as n->0 otherwise: a near-empty
        cell would poison recover_frame)."""
        y = (n / (self.g2d * Te)).clamp_min(1e-8)
        return Te * (y + torch.log1p(-torch.exp(-y)))

    def _fd_jac(self, mu, Te):
        """Analytic thermodynamic Jacobian (dn_dmu, dn_dTe, dEth_dmu, dEth_dTe)."""
        x = mu / Te
        L1 = torch.logaddexp(torch.zeros_like(x), x)
        sig = torch.sigmoid(x)
        n = self.g2d * Te * L1
        Eth = self.Eth_FD(mu, Te)
        return self.g2d * sig, self.g2d * (L1 - x * sig), n, 2.0 * Eth / Te - x * n

    def recover_frame(self, U, Te_guess=None, iters: int = 6):
        """(n,Jx,Jy,E) -> (mu, Te, u).  u,Eth analytic; Te by a fixed Newton on the
        n-constant path Eth_FD(mu_of_nT(n,Te),Te)=Eth (compile-safe, no host sync);
        mu analytic.  U: (...,4)."""
        n = U[..., 0].clamp_min(1e-10 * self.g2d * self.E_F)   # physical floor >> mu_of_nT -inf threshold
        u = U[..., 1:3] / (self.mstar * n[..., None])
        Eth = U[..., 3] - 0.5 * self.mstar * n * (u * u).sum(-1)
        Te = torch.full_like(n, self.T_temp) if Te_guess is None else Te_guess.clone()
        for _ in range(iters):
            mu = self.mu_of_nT(n, Te)
            dn_dmu, dn_dTe, dEth_dmu, dEth_dTe = self._fd_jac(mu, Te)
            g = self.Eth_FD(mu, Te) - Eth
            gp = dEth_dTe - dEth_dmu * (dn_dTe / dn_dmu)          # dEth/dTe at fixed n
            Te = torch.clamp(Te - g / gp, min=1e-6 * self.T_temp)
        return self.mu_of_nT(n, Te), Te, u

    def _dev(self, f):
        """Deviation d = f - f0 reshaped to (..., Nr, N_theta)."""
        Nr, Nth = self.Nr, self.angular.N_theta
        return (f - self.rho0).reshape(*f.shape[:-1], Nr, Nth)

    def shell_moments(self, f, mu, Te):
        """FLAT-measure deviation moments -> stress P (incl. FD core) and 3rd moment
        M3; all (...,).  f: (..., Nk)."""
        d = self._dev(f)
        Jc = self.mstar * Te / self.hbar ** 2
        wk = (self.radial.flat_w[:, None] * self.angular.wphi) * Jc[..., None, None]
        kb = self._kbar(mu, Te)[..., :, None]                    # (...,Nr,1)
        cph, sph = torch.cos(self.angular.theta), torch.sin(self.angular.theta)
        kx, ky, k2 = kb * cph, kb * sph, kb * kb
        wkd = wk * d
        Pxx = self.cnorm * (wkd * kx * kx).sum((-1, -2))
        Pyy = self.cnorm * (wkd * ky * ky).sum((-1, -2))
        Pxy = self.cnorm * (wkd * kx * ky).sum((-1, -2))
        M3x = self.cnorm * (wkd * k2 * kx).sum((-1, -2))
        M3y = self.cnorm * (wkd * k2 * ky).sum((-1, -2))
        trc = (2.0 * self.mstar / self.hbar ** 2) * self.Eth_FD(mu, Te)   # isotropic FD core
        return 0.5 * trc + Pxx, 0.5 * trc + Pyy, Pxy, M3x, M3y

    def assemble_fluxes(self, f, mu, Te, u):
        """Lab fluxes (Fn=n u, Pi, q) -- frame-independent physical tensors -- from
        the shape f and the per-cell frame.  Fn:(...,2) Pi:(...,2,2) q:(...,2)."""
        n = self.n_FD(mu, Te); Eth = self.Eth_FD(mu, Te)
        Pxx, Pyy, Pxy, M3x, M3y = self.shell_moments(f, mu, Te)
        c = self.hbar ** 2 / self.mstar; u2 = (u * u).sum(-1)
        Fn = n[..., None] * u
        Pxx_l = c * Pxx + self.mstar * n * u[..., 0] ** 2
        Pyy_l = c * Pyy + self.mstar * n * u[..., 1] ** 2
        Pxy_l = c * Pxy + self.mstar * n * u[..., 0] * u[..., 1]
        Pi = torch.stack([torch.stack([Pxx_l, Pxy_l], -1),
                          torch.stack([Pxy_l, Pyy_l], -1)], -2)
        Pu = torch.stack([c * (Pxx * u[..., 0] + Pxy * u[..., 1]),
                          c * (Pxy * u[..., 0] + Pyy * u[..., 1])], -1)
        qsh = self.hbar ** 3 / (2 * self.mstar ** 2)
        q = ((Eth + 0.5 * self.mstar * n * u2)[..., None] * u
             + Pu + qsh * torch.stack([M3x, M3y], -1))
        return Fn, Pi, q

    def eq_flux(self, mu, Te, u, nx, ny):
        """Analytic drifted-heated-FD equilibrium lab flux dotted with the face
        normal (nx,ny): (Phi_n, Phi_Jx, Phi_Jy, Phi_E).n^  ->  (...,4).  Closed
        form of assemble_fluxes(rho0, .).n^ (deviation d=0), carrying the filled
        Fermi CORE analytically with NO shell contraction.  Used by the interior
        kinetic flux-vector split as the central equilibrium term."""
        n = self.n_FD(mu, Te); Eth = self.Eth_FD(mu, Te)
        un = u[..., 0] * nx + u[..., 1] * ny
        u2 = (u * u).sum(-1)
        return torch.stack([n * un,
                            Eth * nx + self.mstar * n * u[..., 0] * un,
                            Eth * ny + self.mstar * n * u[..., 1] * un,
                            (2.0 * Eth + 0.5 * self.mstar * n * u2) * un], -1)

    def eq_abs_flux(self, mu, Te, u, nx, ny):
        """Kinetic |v_lab.n^| ABSOLUTE moments of the drifted-heated FD equilibrium
        (Psi_n, Psi_Jx, Psi_Jy, Psi_E) -> (...,4): the flux-vector-split (KFVS-on-
        f0) dissipation that carries the v_F sound characteristics a central
        equilibrium misses.  This is the ONLY interior dissipation (no Rusanov).

        EXACT half-range |v.n^| moments of the filled drifted disk (degenerate T=0
        Fermi sea of radius v_F centred at the drift u), as closed piecewise forms
        in the drift ratio  s = u.n^/v_F.  I0, I1, I02 are the disk integrals of
        |xi+s| against {1, xi, xi^2+eta^2} over the unit disk (xi,eta):

            |s|<1 (subsonic):
              I0  = (2/3) r (2+s^2)           + 2 s asin(s)
              I1  = (s/6) r (5-2 s^2)         + (1/2) asin(s)
              I02 = r (4/5 + s^2/15 + 2 s^4/15) + s asin(s)      r = sqrt(1-s^2)
            |s|>=1 (supersonic, drift-dominated -- v_F cancels analytically):
              I0 = pi|s|,   I1 = (pi/4) sign(s),   I02 = (pi/2)|s|

        The exact form is (i) ALWAYS PSD -- I0>=4/3>0 for every s (the old quartic
        fit crossed zero at s=3.459 -> anti-dissipative negative diffusion), and
        (ii) FINITE as mu->0: v_F is floored, and for |s|>=1 the branch cancels the
        floored v_F exactly (Psi_n->n|u.n^|, Psi_Jn->m* n |u.n^|(u.n^),
        Psi_E->1/2 m* n |u.n^| |u|^2), so the floor value is physically inert.
        Batched over a leading axis."""
        n = self.n_FD(mu, Te)
        # Floor v_F so mu<=0 (a valid hot/non-degenerate state that recover_frame
        # returns) stays finite; only triggers for mu < ~1e-24 E_F (i.e. mu<=0).
        vF = torch.sqrt(torch.clamp(2.0 * mu / self.mstar, min=0.0)).clamp_min(1e-12 * self.vF)
        un = u[..., 0] * nx + u[..., 1] * ny            # u.n^   (normal drift)
        ut = -u[..., 0] * ny + u[..., 1] * nx           # u.t^   (tangential drift)
        u2 = (u * u).sum(-1)
        s = un / vF
        s2 = s * s
        sub = s2 < 1.0
        r = torch.sqrt(torch.clamp(1.0 - s2, min=0.0))          # sqrt(1-s^2); 0 if |s|>=1
        asr = torch.asin(s.clamp(-1.0, 1.0))                    # arcsin s (subsonic branch)
        absS = s.abs()
        sgnS = torch.sign(s)
        I0 = torch.where(sub, (2.0 / 3.0) * r * (2.0 + s2) + 2.0 * s * asr,
                         np.pi * absS)
        I1 = torch.where(sub, (s / 6.0) * r * (5.0 - 2.0 * s2) + 0.5 * asr,
                         (np.pi / 4.0) * sgnS)
        I02 = torch.where(sub, r * (0.8 + s2 / 15.0 + 2.0 * s2 * s2 / 15.0) + s * asr,
                          (np.pi / 2.0) * absS)
        pref = n / np.pi
        Psi_n  = pref * vF * I0
        Psi_Jn = self.mstar * pref * vF * vF * (I1 + s * I0)    # normal-momentum
        Psi_Jt = self.mstar * ut * Psi_n                       # tangential (passive)
        Psi_E  = 0.5 * self.mstar * pref * vF ** 3 * (I02 + 2.0 * s * I1 + (u2 / (vF * vF)) * I0)
        return torch.stack([Psi_n, Psi_Jn * nx - Psi_Jt * ny,
                            Psi_Jn * ny + Psi_Jt * nx, Psi_E], -1)

    def project_moment_free(self, d, mu, Te):
        """Remove the {1, xi', k_bar cos, k_bar sin} components of the deviation d in
        the FLAT-measure shell inner product.  On GL radial + midpoint angular nodes
        the Gram is EXACTLY diagonal, so this is 4 independent scalar projections
        (Jc, wphi cancel).  Nulls (int df, <k_bar>, int eps df) to round-off, any
        drift.  d: (..., Nk)  ->  d_perp: (..., Nk)."""
        Nr, Nth = self.Nr, self.angular.N_theta
        dr = d.reshape(*d.shape[:-1], Nr, Nth)
        w = self.radial.flat_w[:, None].expand(Nr, Nth)          # (Nr,N_theta) full measure
        xi = self.radial.xi[:, None]                             # (Nr,1)
        kb = self._kbar(mu, Te)[..., :, None]                    # (...,Nr,1)
        cph, sph = torch.cos(self.angular.theta), torch.sin(self.angular.theta)

        def strip(cur, B):                                       # remove <B,cur>/<B,B> B
            num = (w * B * cur).sum((-1, -2), keepdim=True)
            den = (w * B * B).sum((-1, -2), keepdim=True).clamp_min(1e-300)
            return cur - (num / den) * B

        dr = strip(dr, torch.ones((), dtype=d.dtype, device=d.device))
        dr = strip(dr, xi)
        dr = strip(dr, kb * cph)
        dr = strip(dr, kb * sph)
        return dr.reshape(d.shape)

    def pauli_reproject(self, f, mu, Te):
        """Pauli clip to [0,1] then re-project the deviation moment-free (restores
        exact consistency; leaves a tiny bounded overshoot)."""
        return self.rho0 + self.project_moment_free(f.clamp(0.0, 1.0) - self.rho0, mu, Te)

    def U_from_frame(self, mu, Te, u):
        """Conserved lab densities (n,Jx,Jy,E) from the analytic frame (f=f0)."""
        n = self.n_FD(mu, Te)
        J = self.mstar * n[..., None] * u
        E = self.Eth_FD(mu, Te) + 0.5 * self.mstar * n * (u * u).sum(-1)
        return torch.cat([n[..., None], J, E[..., None]], -1)

    def moments_of_f(self, f, mu, Te, u):
        """Physical lab (n,Jx,Jy,E) computed FROM the shape f in the given frame --
        the consistency probe.  Equals U_from_frame after project_moment_free."""
        d = self._dev(f)
        Jc = self.mstar * Te / self.hbar ** 2
        wk = (self.radial.flat_w[:, None] * self.angular.wphi) * Jc[..., None, None]
        kb = self._kbar(mu, Te)[..., :, None]
        cph, sph = torch.cos(self.angular.theta), torch.sin(self.angular.theta)
        # Frame energy weight = affine eps_df = mu + Te xi' (== 1/2 hbar^2 kbar^2/m*
        # in the degenerate window, but stays consistent with the {1,xi'} projection
        # basis when heating pushes nodes toward/below the band bottom).
        eps = (mu[..., None] + Te[..., None] * self.radial.xi)[..., :, None]
        n = self.n_FD(mu, Te) + self.cnorm * (wk * d).sum((-1, -2))
        kbx = self.cnorm * (wk * kb * cph * d).sum((-1, -2))
        kby = self.cnorm * (wk * kb * sph * d).sum((-1, -2))
        Eth = self.Eth_FD(mu, Te) + self.cnorm * (wk * eps * d).sum((-1, -2))
        kD = self.mstar * u / self.hbar
        Jx = self.hbar * (kD[..., 0] * n + kbx)
        Jy = self.hbar * (kD[..., 1] * n + kby)
        E = Eth + 0.5 * self.mstar * n * (u * u).sum(-1) + self.hbar * (u[..., 0] * kbx + u[..., 1] * kby)
        return torch.stack([n, Jx, Jy, E], -1)

    # ---- k-space grid-motion transport (14) + volume GCL (18) ----
    def dframe_from_dU(self, dU, mu, Te, u):
        """d_t(mu,Te,k_D) from the marched dU=(dn,dJ,dE) (eqs 165/168/169)."""
        n = self.n_FD(mu, Te)
        dn, dJ, dE = dU[..., 0], dU[..., 1:3], dU[..., 3]
        kD = self.mstar * u / self.hbar
        dkD = dJ / (self.hbar * n[..., None]) - (kD / n[..., None]) * dn[..., None]
        dEth = dE - (u * dJ).sum(-1) + 0.5 * self.mstar * (u * u).sum(-1) * dn
        a, b, c, d = self._fd_jac(mu, Te)
        det = a * d - b * c
        dmu = (d * dn - b * dEth) / det
        dTe = (-c * dn + a * dEth) / det
        return dmu, dTe, dkD

    def shell_velocities(self, mu, Te, u, dmu, dTe, dkD, gmu, gTe, gkD):
        """Grid velocities xidot (165), phidot (166) per (K,Nr,Nθ).  D q = d_t q +
        (v+u).grad_r q ; v = ħ k̄/m* (cosθ,sinθ).  gmu,gTe:(K,2)  gkD:(K,2,2)=d_d(k_D)_i."""
        kb = self._kbar(mu, Te)                                    # (K,Nr)
        cph = torch.cos(self.angular.theta); sph = torch.sin(self.angular.theta)
        vx = (self.hbar / self.mstar) * kb[:, :, None] * cph       # (K,Nr,Nθ)
        vy = (self.hbar / self.mstar) * kb[:, :, None] * sph
        vpx = vx + u[:, 0][:, None, None]; vpy = vy + u[:, 1][:, None, None]

        def D(dq, gq):
            return dq[:, None, None] + vpx * gq[:, 0][:, None, None] + vpy * gq[:, 1][:, None, None]
        Dmu, DTe = D(dmu, gmu), D(dTe, gTe)
        DkDx = D(dkD[:, 0], gkD[:, :, 0]); DkDy = D(dkD[:, 1], gkD[:, :, 1])
        hv_DkD = self.hbar * (vx * DkDx + vy * DkDy)
        xip = self.radial.xi[None, :, None]
        xidot = -(hv_DkD + Dmu + xip * DTe) / Te[:, None, None]
        phidot = -(-sph * DkDx + cph * DkDy) / kb[:, :, None]
        return xidot, phidot

    def kspace_div(self, G, xidot, phidot, glo, ghi):
        """Radial (ξ', GL nodes, midpoint faces, control-vol=flat_w) + angular (φ,
        periodic) conservative divergence of a shell density G=f𝒥.  glo/ghi:(...,1,Nθ)
        core/tail ghosts, or None for zero-gradient (the GCL uniform-f test)."""
        Nr, Nth = self.Nr, self.angular.N_theta
        G = G.reshape(*G.shape[:-1], Nr, Nth)
        if glo is None:
            glo, ghi = G[..., :1, :], G[..., -1:, :]
        Gpad = torch.cat([glo, G, ghi], dim=-2)
        xdp = torch.cat([xidot[..., :1, :], xidot, xidot[..., -1:, :]], dim=-2)
        xdf = 0.5 * (xdp[..., :-1, :] + xdp[..., 1:, :])
        Gup = torch.where(xdf > 0, Gpad[..., :-1, :], Gpad[..., 1:, :])
        Fr = xdf * Gup
        out = -(Fr[..., 1:, :] - Fr[..., :-1, :]) / self.radial.flat_w[:, None]
        pdf = 0.5 * (phidot + torch.roll(phidot, -1, dims=-1))
        Ga = torch.where(pdf > 0, G, torch.roll(G, -1, dims=-1))
        Fa = pdf * Ga
        out = out - (Fa - torch.roll(Fa, 1, dims=-1)) / self.angular.wphi
        return out.reshape(*out.shape[:-2], Nr * Nth)

    def get_contactor(self, n: torch.Tensor, **kwargs) -> Callable:
        return _FermiSurfaceContactor(self, n, **kwargs)

    def get_reflector(self, n: torch.Tensor) -> Callable:
        return _FermiSurfaceReflector(self, n, self.specularity)

    def initialize_fields(self, rho, params, patch_id) -> None:
        pass

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        a = cp_path.attrs
        a["kF"], a["vF"] = self.kF, self.vF
        a["M_theta"], a["Nr"] = self.M_theta, self.Nr
        a["T"], a["xi_max"]   = self.T_temp, self.xi_max
        a["tau_p"]  = (1.0 / self.tau_inv_p)  if self.tau_inv_p  else np.inf
        a["tau_ee"] = (1.0 / self.tau_inv_ee) if self.tau_inv_ee else np.inf
        a["r_c"], a["specularity"] = self.r_c, self.specularity
        return list(a.keys())


# ----------------------------------------------------------------------------
# Contactor: voltage dmu + drift current vD (Dirichlet ghost in delta-k)
# ----------------------------------------------------------------------------
class _FermiSurfaceContactor:
    """Contact distribution constructed in modes, then transformed to delta-k.

    Voltage shift dmu sets the equilibrium-shape (n=0, m=0) mode.  Drift current
    vD into the device sets the (n=0, m=1) cos/sin pair, rotated to the wall
    outward normal.  Higher modes are left at zero.  For Nr=1 the n=0 mode is
    the only radial mode and this matches ``ModalContactor`` identically; for
    Nr>1 the equilibrium-shape part lives entirely in the n=0 radial mode (the
    particle-number null mode of L^(ee)) and the equilibrium derivative w.r.t.
    chemical potential / drift is *exactly* representable there.
    """

    def __init__(
        self, fs: "FermiSurface", n: torch.Tensor, *,
        dmu: float = 0.0, vD: float = 0.0,
    ) -> None:
        n = n.to(rc.device)  # accept normals supplied on any device
        Nsel = n.shape[0]
        # ---- full-f contact: genuine (drifted-heated) Fermi-Dirac ghost ------
        # Build the reservoir distribution DIRECTLY in nodes (0 <= f <= 1):
        #     f_c[r,q] = sigmoid(dmu/T + (k.u_D)/T - xi_r),
        # with drift velocity u_D = -vD * n_hat (vD>0 injects current inward),
        #     (k.u_D)/T = -(kF/(vF T)) |v_r| vD cos(theta_q - phi).
        # At vD=0 this is the isotropic FD  sigmoid(dmu/T - xi_r) -- the reservoir
        # a {I_set, vD=0} current source samples (its level dmu is solved by the
        # geometry layer so the net emitted current hits I_set).
        theta = fs.angular.theta                          # (N_theta,)
        xi_r = fs.radial.xi                               # (Nr,)
        inv_T = 1.0 / fs.T_temp
        phi = torch.atan2(n[:, 1], n[:, 0])               # (Nsel,)
        cos_qmphi = (
            torch.cos(theta)[None, :] * torch.cos(phi)[:, None]
            + torch.sin(theta)[None, :] * torch.sin(phi)[:, None]
        )                                                 # (Nsel, N_theta)
        drift_coeff = -(fs.kF / (fs.vF * fs.T_temp)) * float(vD)
        drift_rq = (drift_coeff * fs.v_speed[None, :, None]
                    * cos_qmphi[:, None, :])              # (Nsel, Nr, N_theta)
        base_arg = (dmu * inv_T) + drift_rq - xi_r[None, :, None]
        self._base_arg = base_arg                         # ghost arg at dmu given
        self._inv_T = inv_T
        f_c = torch.sigmoid(base_arg)                     # (Nsel, Nr, N_theta)
        self.rho_contact = f_c.reshape(Nsel, fs.Nr * fs.angular.N_theta)

    def __call__(self, t: float) -> torch.Tensor:
        return self.rho_contact


# ----------------------------------------------------------------------------
# Reflector: specular block-rotation per harmonic + diffuse (n=0,m=0) refill,
# both done in modes inside this class so the geometry layer just substitutes
# the returned u^P as the Dirichlet ghost.
# ----------------------------------------------------------------------------
class _FermiSurfaceReflector:
    """Boundary reflection for FermiSurface: specular fraction s, diffuse (1-s).

    Specular in modes: identical to ``ModalReflector._specular`` -- flip the
    sin(m theta) coefficients, then rotate harmonic m by ``m * (2 phi + pi)`` --
    applied independently on each radial slice (radial index commutes with
    angular rotation).  Exact at any wall angle.

    Diffuse in modes: outgoing is set to ``D`` in the (n=0, m=0) mode only,
    everything else zero, where ``D`` is fixed by zero net normal mass flux:
        D = sum_q (v_q . n)_+  u^M_{r=0}(theta_q)  /  sum_q (v_q . n)_- (with sign).
    Here ``u^M_{r=0}(theta_q) = sum_r T_to_radial[0, r] u^M(k_r, theta_q)`` is
    the n=0 radial projection of u^M -- only this projection contributes to the
    mass-flux balance (higher radial modes carry no mass).  For ``Nr = 1`` this
    is just ``u^M(theta_q)`` and the formula reduces to the scalar one used by
    ``_dg_torch`` for discrete-ordinate single_band materials.
    """

    def __init__(
        self, fs: "FermiSurface", n: torch.Tensor, specularity: float,
    ) -> None:
        n = n.to(rc.device)  # accept normals supplied on any device
        self.fs = fs
        self.s = float(specularity)
        self.M_theta = fs.M_theta
        self.Nr = fs.Nr
        self.dim_theta = fs.angular.dim
        self.N_theta = fs.angular.N_theta
        # outward-normal angle and the rotation angle 2 phi + pi
        self.phi   = torch.atan2(n[:, 1], n[:, 0])
        self.angle = 2.0 * self.phi + np.pi
        # per-(wall, ordinate) v dot n (same for every radial point)
        theta = fs.angular.theta
        v_dot_n = fs.vF * (
            n[:, 0:1] * torch.cos(theta)[None, :] +
            n[:, 1:2] * torch.sin(theta)[None, :]
        )                                                    # (Nsel, N_theta)
        self.adn_pos = v_dot_n.clamp(min=0.0)
        self.adn_neg = v_dot_n.clamp(max=0.0)
        self.w_in    = (-self.adn_neg).sum(-1).clamp(min=1e-300)
        # n=0 radial projector  T_to_radial[0, :] (length Nr); for Nr=1 this is [1.0]
        self.T_to_rad_0 = fs.radial.T_to_modes[0, :]         # (Nr,)
        # n=0 radial basis function psi_0(xi_r) = T_from_radial[:, 0] (length Nr)
        self.psi_0 = fs.radial.T_from_modes[:, 0]            # (Nr,)
        # ---- full-f diffuse geometry (genuine Fermi-Dirac re-emit) ----------
        # For full f the diffuse wall re-emits an ISOTROPIC genuine Fermi-Dirac
        #     f_w(xi_r) = sigmoid(mu_tilde - xi_r),   mu_tilde = dmu_w / T
        # with a SINGLE mu_tilde per wall face fixed by zero net normal mass
        # flux.  The discrete normal mass-flux carrier per (wall, r, ordinate)
        # is  mflux = (flat_w[r] * |v_r| / N_theta) * (k_hat . n),  splitting
        # into outflow (k_hat.n > 0, carried by the interior trace) and inflow
        # (k_hat.n < 0, carried by the ghost).  A_r = sum_{inflow} mflux (< 0)
        # is the coefficient multiplying f_w[r] in the inflow mass flux.
        # The measure is the FLAT phase-space weight flat_w (NOT the sech^2 quad_w):
        # the physical particle flux -- and the flat-measure moment the moving-frame
        # U-march books at the wall -- is the flat one, so balancing the diffuse
        # re-emit here in flat_w makes the wall's net normal mass flux round-off
        # (a sech^2 balance left an O(1e-4 n vF) flat-measure leak; see final_review2).
        khat_dot_n = (
            n[:, 0:1] * torch.cos(theta)[None, :] +
            n[:, 1:2] * torch.sin(theta)[None, :]
        )                                                # (Nsel, N_theta)
        mcoef_r = fs.radial.flat_w * fs.v_speed / self.N_theta   # (Nr,)
        mflux = mcoef_r[None, :, None] * khat_dot_n[:, None, :]   # (Nsel,Nr,Nth)
        pos_mask = (khat_dot_n > 0.0)[:, None, :]                 # (Nsel,1,Nth)
        neg_mask = (khat_dot_n < 0.0)[:, None, :]
        self.mf_pos = mflux * pos_mask                           # (Nsel,Nr,Nth)
        self.mf_neg = mflux * neg_mask
        self.A_r = self.mf_neg.sum(-1)                           # (Nsel,Nr) < 0
        self.A_tot = self.A_r.sum(-1).clamp(max=-1e-300)         # (Nsel,) < 0
        self.xi_r = fs.radial.xi                                 # (Nr,)
        # Bracket for the mu_tilde bisection (below/above the full xi spread;
        # +-40 comfortably saturates every sigmoid).  Precomputed 0-d tensors.
        self.mu_lo = fs.radial.xi.min() - 40.0                   # 0-dim
        self.mu_hi = fs.radial.xi.max() + 40.0                   # 0-dim

    # ---- specular: block-rotation R(phi) per harmonic, independent in radial ----
    def _specular_modal(self, a_modal: torch.Tensor) -> torch.Tensor:
        shape_in = a_modal.shape
        a4 = a_modal.reshape(*shape_in[:-1], self.Nr, self.dim_theta)
        out = a4.clone()                                     # (n=0, m=0) untouched
        angle = self.angle                                   # (Nsel,)
        for m in range(1, self.M_theta + 1):
            c   = torch.cos(m * angle)
            sn  = torch.sin(m * angle)
            a_c = a4[..., 2 * m - 1]
            a_s = -a4[..., 2 * m]                            # flip sin coeff first
            # rotate (a_c, a_s) by m*angle:  (c*a - sn*b, sn*a + c*b)
            out[..., 2 * m - 1] = c[..., None] * a_c - sn[..., None] * a_s
            out[..., 2 * m]     = sn[..., None] * a_c + c[..., None] * a_s
        return out.reshape(shape_in)

    # ---- (D_n, T_n) 2x2 wall correction: exact discrete mass + tangential mom. ----
    def _dt_added(self, uM_dk: torch.Tensor, spec_dk: torch.Tensor,
                  s: float) -> torch.Tensor:
        """Additive outgoing correction that enforces discrete mass AND
        tangential-momentum conservation at the wall to machine precision.

        Returns ``u_added(r, q) = sum_n psi_n(xi_r) [D_n + T_n sin(theta_q-phi)]``
        (reshaped to ``uM_dk.shape``), where per radial mode n the 2x2 system

            [-w_in     beta    ] [D_n]   [-F_M_n     ]
            [vF beta   vF gamma] [T_n] = [-s * F_T_n ]

        is solved by Cramer's rule.  ``D_n`` fixes the (n, m=0) mass-like mode and
        ``T_n`` the (n, m=1 tangential) mode of the outgoing addition; the blocks
        decouple per n because the velocity is radial-index-independent.  The RHS
        mixing ``s`` is a specularity weight; the full-f reflector calls this with
        ``s = 1`` to impose the *pure-specular* balance (F_M_n = F_out + F_in_spec,
        F_T_n = F_out_tang + F_in_spec_tang) so the specular reflection alone
        conserves mass and tangential momentum.  Since the isotropic equilibrium
        f0 carries zero net normal and tangential flux, the correction computed
        from the full f equals the one carried by the deviation about f0."""
        shape_in = uM_dk.shape
        uM4 = uM_dk.reshape(*shape_in[:-1], self.Nr, self.N_theta)
        sp4 = spec_dk.reshape(*shape_in[:-1], self.Nr, self.N_theta)
        # radial-n projection: u^M_n(q) = sum_r T_to_radial[n, r] u^M(r, q)
        T_to_r   = self.fs.radial.T_to_modes                # (Nr, Nr)
        T_from_r = self.fs.radial.T_from_modes              # (Nr, Nr)
        uM_n_q = torch.einsum("nr,...arq->...anq", T_to_r, uM4)
        sp_n_q = torch.einsum("nr,...arq->...anq", T_to_r, sp4)
        # angular basis sin(theta_q - phi) per wall node
        sin_q = (torch.cos(self.phi)[:, None] * torch.sin(self.fs.angular.theta)[None, :]
                 - torch.sin(self.phi)[:, None] * torch.cos(self.fs.angular.theta)[None, :])
        # 2x2 wall-geometry coefficients (Nsel,) -- same for every n
        beta  = (self.adn_neg * sin_q).sum(-1)
        gamma = (self.adn_neg * sin_q ** 2).sum(-1)
        # discrete fluxes per (Nsel, Nr)
        adn_pos = self.adn_pos[:, None, :]
        adn_neg = self.adn_neg[:, None, :]
        sin_b   = sin_q[:, None, :]
        vF = self.fs.vF
        F_out_mass_n     = (adn_pos * uM_n_q).sum(-1)
        F_in_spec_mass_n = (adn_neg * sp_n_q).sum(-1)
        F_out_tang_n     = vF * (adn_pos * sin_b * uM_n_q).sum(-1)
        F_in_spec_tang_n = vF * (adn_neg * sin_b * sp_n_q).sum(-1)
        F_M_n = F_out_mass_n + s * F_in_spec_mass_n
        F_T_n = F_out_tang_n + F_in_spec_tang_n
        # 2x2 Cramer per (Nsel), broadcast over Nr (and any leading batch).
        det = -vF * (self.w_in * gamma + beta * beta)        # (Nsel,)  < 0
        det_b = det[:, None]
        b1 = -F_M_n
        b2 = -s * F_T_n
        D = (b1 * (vF * gamma)[:, None] - b2 * beta[:, None]) / det_b
        T = ((-self.w_in)[:, None] * b2 - (vF * beta)[:, None] * b1) / det_b
        # u_added(r, q) = sum_n psi_n(r) [D_n + T_n sin_q],  via T_from_modes.
        D_r = torch.einsum("rn,...an->...ar", T_from_r, D)
        T_r = torch.einsum("rn,...an->...ar", T_from_r, T)
        u_added = D_r.unsqueeze(-1) + T_r.unsqueeze(-1) * sin_q.unsqueeze(-2)
        return u_added.reshape(shape_in)

    # ---- full-f reflection: specular (exact on full f) + genuine-FD diffuse ----
    def _call_full_f(self, uM_dk: torch.Tensor) -> torch.Tensor:
        """Reflect the FULL distribution f (0<=f<=1) at the wall.

        Specular is exact on full f: the modal rotation leaves the isotropic
        (n=0, m=0) part -- which carries f0 -- untouched and rotates m>=1, so
        R(f0 + df) = f0 + R(df).  The finite-N_theta quadrature of the reflected
        trace leaks a little mass at curved walls and tangential momentum at
        oblique walls; the (D_n, T_n) 2x2 correction (pure-specular s=1 balance)
        removes both to machine precision -- f0 carries zero net normal and
        tangential flux, so the correction on full f equals that on df and is a
        tiny additive term that leaves f in [0,1].  Diffuse re-emits an isotropic
        genuine Fermi-Dirac at a single chemical potential mu_tilde per wall
        face, fixed by zero net normal mass flux (exact to machine precision).
        """
        spec_modal = self._specular_modal(self.fs.to_modes(uM_dk))
        spec_dk = self.fs.from_modes(spec_modal)              # = f0 + R(df)
        spec_dk = spec_dk + self._dt_added(uM_dk, spec_dk, 1.0)  # mass+tang exact
        s = self.s
        if s >= 1.0:
            return spec_dk
        shape_in = uM_dk.shape
        uM4 = uM_dk.reshape(*shape_in[:-1], self.Nr, self.N_theta)
        sp4 = spec_dk.reshape(*shape_in[:-1], self.Nr, self.N_theta)
        # Net normal mass flux = outflow(interior) + s*inflow(spec) + (1-s)*inflow(f_w)
        # must vanish  =>  sum_r f_w[r] * A_r = RHS,   RHS = -(J_out + s S_spec)/(1-s)
        J_out  = (self.mf_pos * uM4).sum((-1, -2))            # (...,Nsel)
        S_spec = (self.mf_neg * sp4).sum((-1, -2))            # (...,Nsel)
        RHS = -(J_out + s * S_spec) / (1.0 - s)               # (...,Nsel)
        # Solve  sum_r sigmoid(mu_tilde - xi_r) * A_r = RHS  for the single
        # per-face chemical potential mu_tilde.  h(mu) = LHS - RHS is monotonically
        # DECREASING in mu (all A_r <= 0) from +|RHS| (mu -> -inf) to A_tot - RHS
        # (mu -> +inf); the physical RHS in [A_tot, 0] is bracketed by
        # [xi_min - 40, xi_max + 40].  Bisection is used rather than Newton because
        # a fixed-iteration Newton diverges when some radial nodes sit below the
        # band bottom (|v_r| = 0 -> A_r = 0): the objective is then flat in mu over
        # those nodes and the flat-f_w init overshoots to +-inf.  60 halvings pin
        # mu_tilde to ~1e-16; bisection also self-saturates if RHS is (marginally)
        # outside the bracket, so the ghost stays Pauli-bounded unconditionally.
        lo = torch.zeros_like(RHS) + self.mu_lo               # (...,Nsel)
        hi = torch.zeros_like(RHS) + self.mu_hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            h = (torch.sigmoid(mid[..., None] - self.xi_r) * self.A_r).sum(-1) - RHS
            take_upper = h > 0.0                              # root lies at larger mu
            lo = torch.where(take_upper, mid, lo)
            hi = torch.where(take_upper, hi, mid)
        mu = 0.5 * (lo + hi)                                  # (...,Nsel)
        fw = torch.sigmoid(mu[..., None] - self.xi_r).clamp(0.0, 1.0)  # (...,Nsel,Nr)
        diff = fw.unsqueeze(-1).expand(*fw.shape, self.N_theta)        # isotropic
        ghost = s * sp4 + (1.0 - s) * diff
        return ghost.reshape(shape_in)

    # ---- main call ----
    def __call__(self, uM_dk: torch.Tensor) -> torch.Tensor:
        return self._call_full_f(uM_dk)
