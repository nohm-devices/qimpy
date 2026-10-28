from __future__ import annotations
from typing import Callable, Optional, Union
from dataclasses import dataclass
import os

import numpy as np
import torch

from qimpy import log, rc, TreeNode
from qimpy.mpi import ProcessGrid, TaskDivision
from qimpy.profiler import stopwatch
from qimpy.io import CheckpointPath, CheckpointContext, InvalidInputException
from .scattering import EEScattering
from .._material import Material
from ._representation import DeltaK, KRepresentation
from ._cartesian import Cartesian


class FermiSurface(Material):
    """Model isotropic 2D electron-gas material.

    Supports Brillouin zone sampling in cylindrical coordinates in the vicinity of
    the Fermi circle (`delta_k`, default), or Cartesian (`cartesian`) for large drift.
    """

    kF: float  #: Fermi wave-vector
    vF: float  #: Fermi velocity
    m_star: float  #: Effective mass
    mu: float  #: Chemical potential
    n_theta: int  #: Number of azimuthal k-points and angular modes
    n_r: int  #: Number of radial k-points and modes
    T: float  #: Temperature
    xi_max: float  #: Maximum (E - Ef)/T sampled radially
    tau_inv_p: float  #: Inverse momentum-relaxation time
    tau_inv_ee: float  #: Inverse momentum-conserving e-e scattering time
    r_c: float  #: Cyclotron radius, finite if magentic field present
    specularity: float  #: Surface specularity
    omega_c: float  #: Fermi circle rotation frequnecy due to magnetic field
    angular: AngularBasis  #: angular basis set by `n_theta`
    radial: RadialBasis  #: radial basis set by `n_r` and `xi_max`.
    representation: KRepresentation  #: underlying Brillouin-zone sampling method

    def __init__(
        self,
        *,
        kF: float,
        vF: float,
        n_theta: int,
        n_r: int = 1,
        T: float = 1.0,
        xi_max: float = 6.0,
        tau_p: float = np.inf,
        tau_ee: float = np.inf,
        r_c: float = np.inf,
        specularity: float = 1.0,
        delta_k: Optional[Union[dict, DeltaK]] = None,
        cartesian: Optional[Union[dict, Cartesian]] = None,
        ee_scattering: Optional[Union[EEScattering, dict]] = None,
        residual_damping: bool = False,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        super().__init__()
        self.kF = kF
        self.vF = vF
        self.m_star = kF / vF
        self.mu = 0.5 * kF * vF  # E_F (degenerate)
        if n_theta % 2:
            raise InvalidInputException(f"{n_theta=} is not even")
        self.n_theta = n_theta
        self.n_r = n_r
        self.T = T
        self.xi_max = xi_max
        self.r_c = r_c
        self.tau_inv_p = 1.0 / tau_p
        self.tau_inv_ee = 1.0 / tau_ee
        self.specularity = specularity
        self.omega_c = (vF / r_c) if np.isfinite(r_c) else 0.0

        # modal bases (shared by both representations and the e-e operator)
        self.angular = AngularBasis(n_theta)
        self.radial = RadialBasis(n_r, T=T, xi_max=xi_max)

        # k-representation (builds the grid; delta-k is the default first option)
        self.add_child_one_of(
            "representation",
            checkpoint_in,
            TreeNode.ChildOptions(
                "delta-k",
                DeltaK,
                delta_k,
                fermi_surface=self,
                process_grid=process_grid,
            ),
            TreeNode.ChildOptions(
                "cartesian",
                Cartesian,
                cartesian,
                fermi_surface=self,
                process_grid=process_grid,
            ),
            have_default=True,
        )
        rep = self.representation
        self.initialize(
            wk=rep.wk, nk=rep.Nk, n_bands=1, n_dim=2, process_grid=process_grid
        )
        if self.group.size() > 1:
            raise InvalidInputException(
                "FermiSurface couples k-channels; the k process-grid dimension "
                "must be 1 (parallelize over space)."
            )
        self.v = rep.v  # transport velocity

        # Phenomenological per-mode collision rates in flattened (n_r, m) ordering
        angular = np.zeros(self.n_theta)
        m_max = self.angular.m_max
        angular[1:] = self.tau_inv_p + self.tau_inv_ee  # all non-zero modes
        angular[1] = angular[m_max + 1] = self.tau_inv_p  # net momentum modes (m = 1)
        radial = np.zeros(n_r)
        radial[1:] = self.tau_inv_ee
        rates = angular[None, :] + radial[:, None]
        rates[0, 0] = 0.0  # particle conservation
        self.rates_modal = rc.from_np(rates.flatten())

        # microscopic e-e collisions (replaces the tau_ee placeholder)
        local_te = None
        if isinstance(ee_scattering, dict):
            ee_scattering = dict(ee_scattering)
            local_te = ee_scattering.pop("local_te", None)
        self.ee_scattering = None
        if (ee_scattering is not None) or checkpoint_in.member("ee_scattering"):
            if np.isfinite(tau_ee):
                raise InvalidInputException(
                    "Specify either phenomenological tau_ee or microscopic"
                    " ee_scattering, not both"
                )
            self.add_child(
                "ee_scattering",
                EEScattering,
                ee_scattering,
                checkpoint_in,
                fermi_surface=self,
            )
        # Exact local-T_e operator ensemble (see the factorization derivation):
        # each degree-d block is EXACTLY C^(d) = T_e^2 T^(1-d) M_d(t_e) with
        # M_d a smooth matrix function of t = T_e/E_F alone -- so tabulate
        # complete operators at Chebyshev nodes T_i = fac_i * T and evaluate
        # per cell by barycentric interpolation with the analytic prefactors.
        # Without the table, a supplied T_e falls back to the leading-order
        # uniform (T_e/T)^2 rescale.
        self._ee_ensemble = None
        if local_te is not None:
            if not isinstance(ee_scattering, dict):
                raise InvalidInputException(
                    "local_te requires ee_scattering given as parameters"
                )
            n_nodes = int(local_te.get("n_nodes", 6))
            fmin = float(local_te.get("te_fac_min", 0.5))
            fmax = float(local_te.get("te_fac_max", 3.0))
            j = np.arange(n_nodes)
            xj = np.cos(np.pi * (2 * j + 1) / (2 * n_nodes))  # Chebyshev
            fac = 0.5 * (fmin + fmax) + 0.5 * (fmax - fmin) * xj
            wj = (-1.0) ** j * np.sin(np.pi * (2 * j + 1) / (2 * n_nodes))
            self._ee_fac = torch.as_tensor(fac, dtype=torch.float64, device=rc.device)
            self._ee_bary_w = torch.as_tensor(wj, dtype=torch.float64, device=rc.device)
            self._ee_fac_lim = (fmin, fmax)
            self._ee_ensemble = [
                EEScattering(fermi_surface=self, T_build=self.T * f, **ee_scattering)
                for f in fac
            ]

        # ---- residual (unresolved-mode) closure rate ------------------------
        self._gamma_res_ee = self._gamma_res_phen = 0.0
        if residual_damping:
            self._set_residual_rate()

    def _set_residual_rate(self) -> None:
        """Closure rate for the modes the projection does not retain.

        The Cartesian k-grid carries angular content up to its own Nyquist
        ``M_Nyq ~ pi kF/dk`` (~199 at production resolution), but the collision
        is evaluated in a basis truncated at ``M_theta``.  The increment the
        operator returns lies entirely in that retained span, so it can neither
        damp nor remove the rest: without a closure the unresolved harmonics
        relax at rate ZERO instead of ``gamma_m``, and since their true
        collisional lifetime (~0.13 transits) is far shorter than the ballistic
        escape (~1 transit) the device looks spuriously ballistic in exactly
        its sharpest angular structure.

        The rate used is the one at the TOP RETAINED HARMONIC -- a zeroth-order
        hold on the rate spectrum, exact through ``M_theta`` then flat.  Chosen
        over any amplitude-weighted average of the discarded band because:

        * ``gamma_m`` is monotone increasing over the EVEN harmonics (verified
          in closed form out to m=420, well past any grid Nyquist), so this is
          a strict LOWER bound on every discarded even rate and cannot
          over-damp the even angular band.  ODD harmonics are excluded from the
          closure entirely -- they are gated to zero at leading order and are
          the long-lived tomographic modes -- see the parity split in
          :meth:`Cartesian.apply_collision`;
        * it is refinement-consistent: raising ``M_theta`` both raises the rate
          toward truth and shrinks the residual band, so M is a clean monotone
          convergence knob with no free constant left over;
        * it introduces no tunable parameter.

        The even spectrum is logarithmic (gamma_198/gamma_48 = 1.36), which is
        what makes a zeroth-order hold adequate; it would be a poor closure for
        a rate growing like m^2.

        CAVEAT (angular only).  The residual also contains RADIAL content
        beyond ``Nr`` at retained even harmonics, which this same rate damps.
        Its true rate at m = 2 is ~gamma_2, so that slice is over-damped by
        roughly gamma_{m_top}/gamma_2 (~4x at M = 48).  The slice is small --
        the hybrid basis reaches 1.7e-3 completeness by Nr = 8 -- but the
        "cannot over-damp" statement above is about the angular band only.

        Split into the microscopic e-e part (rescaled per cell by T_e, like
        every other e-e rate) and the phenomenological part (constant, matching
        how tau_p/tau_ee are handled in :meth:`_modal_collision`).
        """
        m_max = self.angular.m_max
        # The rate must be read at an EVEN harmonic.  Odd harmonics are gated to
        # zero at leading order (K_m = 0 for odd m, _kernels.K_table) and are
        # only weakly relaxed by the exact operator, so an odd M_theta would
        # read gamma ~ 0 and silently disable the closure.
        m_top = m_max - (m_max % 2)
        if m_top < 2:
            raise InvalidInputException(
                "residual_damping requires M_theta >= 2 (the closure rate is "
                "read at the top EVEN harmonic; odd harmonics do not relax at "
                "leading order)"
            )
        if self.n_r < 2:
            raise InvalidInputException(
                "residual_damping requires Nr >= 2: at Nr = 1 the radial span "
                "is {1}, so the ENERGY shape xi itself falls in the residual "
                "and would be damped at the m = M rate, then re-injected by "
                "the conservation projector -- a systematic distortion, not a "
                "small one.  (on_shell=True forces Nr = 1.)"
            )
        if not isinstance(self.representation, Cartesian):
            raise InvalidInputException(
                "residual_damping is implemented only for the Cartesian "
                "representation; the delta-k representation stores the state "
                "modally, where truncation genuinely removes the modes."
            )
        c_top = m_top  # cosine index corresponding to m_top
        # Phenomenological: diagonal, take the slowest radial channel at m_top.
        rates = self.rates_modal.reshape(self.n_r, self.n_theta)
        self._gamma_res_phen = float(rates[:, c_top].min())
        # Microscopic: L_coeff is block-diagonal in the harmonic and a dense
        # (Nr, Nr) matrix over radial modes, in decay form (a_dot = -L @ a).
        # The slowest-decaying radial channel is the smallest eigenvalue of its
        # symmetric part -- the correct dissipativity bound, and it keeps the
        # "never over-damps" guarantee without assuming L is exactly symmetric.
        if self.ee_scattering is not None:
            L = self.ee_scattering.L_coeff[c_top].to(torch.float64).cpu()
            ev = torch.linalg.eigvalsh(0.5 * (L + L.transpose(-1, -2)))
            self._gamma_res_ee = max(float(ev.min()), 0.0)
            g_surf = float(L[0, 0])  # surface-mode rate
            if g_surf > 0.0 and self._gamma_res_ee < 0.1 * g_surf:
                log.warning(
                    f"Residual closure rate {self._gamma_res_ee:.3g} is far "
                    f"below the surface-mode rate {g_surf:.3g} at m={m_top}: a "
                    "radial channel is nearly null there, so the closure will "
                    "be very weak.  Check M_theta / Nr."
                )
        log.info(
            f"Residual closure at m > {m_max}: gamma_res = "
            f"{self._gamma_res_ee + self._gamma_res_phen:.4g} "
            f"(e-e {self._gamma_res_ee:.4g} + phenomenological "
            f"{self._gamma_res_phen:.4g}), read at m={m_top}"
        )

    def gamma_residual(self, te: Optional[torch.Tensor] = None):
        """Per-cell residual closure rate, or None when the closure is off.

        The e-e part carries the same leading ``(T_e/T)^2`` scaling as the
        modal rates it extrapolates; the phenomenological part is a user-set
        constant and is not rescaled.

        With a ``local_te`` ensemble the exact rate is below this leading form
        (the NLO correction is linear in T_e/E_F with negative coefficients,
        ~-7% at T_e ~ 2T), so in hot cells the rescale can overshoot the m=M
        rate by a few percent.  That does not break the closure's lower-bound
        property against the band it stands in for: 1.07 x gamma_M = 4.25 is
        still well under gamma at the grid Nyquist (5.40, in units of gamma_2).
        """
        if self._gamma_res_ee == 0.0 and self._gamma_res_phen == 0.0:
            return None
        if te is None:
            return self._gamma_res_ee + self._gamma_res_phen
        s2 = (te / self.T) ** 2
        return self._gamma_res_ee * s2 + self._gamma_res_phen

    def __getattr__(self, name):
        # Expose the modal transforms only when the representation has them.
        if name in ("to_modes", "from_modes"):
            rep = self.__dict__.get("representation")
            if rep is not None and hasattr(rep, name):
                return getattr(rep, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def constrain(self, rho: torch.Tensor) -> None:
        if (cf := self.representation.constraint_factors) is not None:
            A, B = cf
            rho -= (rho @ A) @ B.T

    @property
    def transport_velocity(self) -> torch.Tensor:
        return self.v

    # ---- the representation-agnostic modal collision (the physics) ----
    def _ee_local_te(self, a: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        """EXACT local-T_e e-e term via the operator ensemble.

        Factorization (derived + agent-verified): each degree-d block is
        C^(d)(T_e) = T_e^2 T^(1-d) M_d(t_e), M_d smooth in t_e = T_e/E_F only.
        A node operator built at T_i carries T_i^2 T_i^(1-d) M_d(t_i) in its own
        convention, so its contribution needs the per-degree factor
        q_d = (T_e/T_i)^2 (T_i/T)^(d-1).  All three degrees are obtained from
        ONE a_dot call per node via homogeneity: with lam_i = T_i/T,
        NL_i(lam a) = lam^2 Q_i + lam^3 C_i, so
        mu_i NL_i(lam_i a) with mu_i = (T_e/T_i)^2 (T/T_i) gives exactly
        q_2 Q_i + q_3 C_i; the linear part is scaled by (T_e/T_i)^2 directly.
        Barycentric-Chebyshev weights in t interpolate M_d spectrally.
        (Per-node null projectors use their own sqrt(1+t_i xi) momentum
        covector; the O(t) covector mismatch of the mixture is absorbed by the
        representation's exact conservation projection downstream.)"""
        T = self.T
        x = (te.to(torch.float64) / T).clamp(*self._ee_fac_lim)  # (cells,)
        d = x[:, None] - self._ee_fac[None, :]  # (cells, N)
        exact = d.abs() < 1e-12
        d = torch.where(exact, torch.ones_like(d), d)
        w = self._ee_bary_w[None, :] / d
        w = torch.where(exact.any(-1, keepdim=True), exact.to(w.dtype), w)
        w = w / w.sum(-1, keepdim=True)  # (cells, N)
        a4 = a.reshape(*a.shape[:-1], self.n_r, self.n_theta)
        out = torch.zeros_like(a)
        for i, ee in enumerate(self._ee_ensemble):
            Ti = float(self._ee_fac[i]) * T
            lin_i = -torch.einsum("cij,...jc->...ic", ee.L_coeff, a4).reshape(a.shape)
            s2 = ((te.to(a.dtype) / Ti) ** 2).unsqueeze(-1)  # (cells, 1)
            term = s2 * lin_i
            if ee.nonlinear:
                lam = Ti / T
                nl_i = ee.a_dot(lam * a) - lam * lin_i  # lam^2 Q + lam^3 C
                mu = s2 * (T / Ti)
                term = term + mu * nl_i
            out = out + w[:, i : i + 1].to(a.dtype) * term
        return out

    def _modal_collision(
        self, a: torch.Tensor, te: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Modal collision.  ``te`` (optional, per-cell) = local electron
        temperature: with a local_te ensemble the microscopic e-e term is
        evaluated EXACTLY at T_e (analytic prefactors x Chebyshev-tabulated
        shape); without one it falls back to the leading-order uniform
        (T_e/T)^2 rescale (all blocks alike; the PH-forbidden Q sector's
        (T_e/T)^3 is O(T/E_F)-small).  Phenomenological tau_p/tau_ee are
        user-set constants and are NOT rescaled; cyclotron is magnetic."""
        a_dot = -self.rates_modal * a
        if self.ee_scattering is not None:
            if te is not None and self._ee_ensemble is not None:
                a_dot = a_dot + self._ee_local_te(a, te)
            else:
                ee_term = self.ee_scattering.a_dot(a)
                if te is not None:
                    s2 = (te.to(ee_term.dtype) / self.T).unsqueeze(-1) ** 2
                    ee_term = ee_term * s2
                a_dot = a_dot + ee_term
        if self.omega_c:
            a4 = a.reshape(*a.shape[:-1], self.n_r, self.n_theta)
            Ga4 = torch.einsum("dc,...nc->...nd", self.angular.derivative, a4)
            a_dot = a_dot + self.omega_c * Ga4.reshape(*a.shape)
        return a_dot

    def gamma_max(self) -> float:
        """Largest linear collision rate (a.u.).

        Sets the collision sub-step budget for operator splitting: an explicit
        sub-flow needs ``gamma_max * dt_coll`` bounded.  NOTE this is the
        fastest eigenvalue over ALL retained modes, which is much larger than
        1/tau_ee read off the m=2 shear mode -- do not size a sub-step from
        l_ee.  Cached; the blocks are tiny (N_r x N_r per angular index).
        """
        if getattr(self, "_gamma_max", None) is None:
            g = float(self.rates_modal.max())
            ee = self.ee_scattering
            if ee is not None:
                L = ee.L_coeff
                Lsym = 0.5 * (L + L.transpose(-1, -2))
                g = max(g, float(torch.linalg.eigvalsh(Lsym).max()))
            gr = self.gamma_residual()
            if gr is not None:
                g = max(g, float(torch.as_tensor(gr).max()))
            self._gamma_max = g
        return self._gamma_max

    @stopwatch
    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        if (
            self.rates_modal.abs().sum() == 0
            and self.omega_c == 0.0
            and self.ee_scattering is None
        ):
            return torch.zeros_like(rho)  # ballistic, no field
        return self.representation.apply_collision(rho, self._modal_collision)

    # ---- observables / boundaries: delegate to the representation ----
    def get_observable_names(self) -> list[str]:
        return self.representation.get_observable_names()

    def get_observables(self, t: float) -> torch.Tensor:
        return self.representation.get_observables()

    def get_flux_indices(self) -> list[int]:
        return self.representation.get_flux_indices()

    def measure_observables(self, rho: torch.Tensor, t: float) -> torch.Tensor:
        try:
            return self.representation.measure_observables(rho)  # if overridden
        except NotImplementedError:
            return super().measure_observables(rho, t)  # base: linear measurments

    def get_contactor(self, n: torch.Tensor, **kwargs) -> Callable:
        return self.representation.get_contactor(n, **kwargs)

    def get_reflector(self, n: torch.Tensor) -> Callable:
        rep = self.representation
        return wrap_dense_reflector(rep.get_reflector(n), n, rep.Nk)

    def initialize_fields(self, rho, params, patch_id) -> None:
        pass

    def update_stash(self, rho: torch.Tensor, r_division: TaskDivision) -> None:
        if self.ee_scattering is not None:
            self.ee_scattering.update_stash(rho, r_division)

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        a = cp_path.attrs
        a["kF"], a["vF"] = self.kF, self.vF
        a["M_theta"], a["Nr"] = self.n_theta, self.n_r
        a["T"], a["xi_max"] = self.T, self.xi_max
        a["tau_p"] = (1.0 / self.tau_inv_p) if self.tau_inv_p else np.inf
        a["tau_ee"] = (1.0 / self.tau_inv_ee) if self.tau_inv_ee else np.inf
        a["r_c"], a["specularity"] = self.r_c, self.specularity
        a["residual_damping"] = bool(self._gamma_res_ee or self._gamma_res_phen)
        return list(a.keys())


class AngularBasis:
    """Real Fourier basis for azimuthal angle.
    Modes are cosine for m = 0, 1, ..., m_max, followed by sine for 1, ..., m_max - 1,
    where m_max = n_theta / 2 is the magnitude of m of the Nyquist component.
    """

    n_theta: int  #: Number of angular nodes and modes (must be even)
    theta: torch.Tensor  #: Evenly spaced quadrature nodes on [0, 2 pi)
    from_modes: torch.Tensor  #: Transform to modes (modes in columns)
    to_modes: torch.Tensor  #: Transform to modes (modes in rows)
    derivative: torch.Tensor  #: Derivative operator in modal basis.

    def __init__(self, n_theta: int) -> None:
        assert n_theta % 2 == 0
        self.n_theta = n_theta
        self.theta = torch.arange(n_theta) * (2 * np.pi / n_theta)
        self.from_modes = self.get_transform(self.theta)

        # Inverse transform:
        m_max = self.m_max
        self.to_modes = self.from_modes.T * (1 / m_max)  # cos, sin normalized
        self.to_modes[0] *= 0.5  # correct zero frequency normalization
        self.to_modes[m_max] *= 0.5  # correct Nyquist frequency normalization

        # Initialize derivative operator:
        self.derivative = torch.zeros_like(self.from_modes)
        m = torch.arange(m_max, dtype=self.derivative.dtype, device=rc.device)
        self.derivative.diagonal(m_max)[:] = -m
        self.derivative.diagonal(-m_max)[:] = m

    @property
    def m_max(self) -> int:
        return self.n_theta // 2

    def get_transform(self, theta: torch.Tensor) -> torch.Tensor:
        m_max = self.m_max
        m_theta = theta[..., None] * torch.arange(m_max + 1, device=rc.device)
        return torch.cat((m_theta.cos(), m_theta[..., 1:m_max].sin()), dim=-1)


class RadialBasis:
    """Radial basis orthonormal with Fermi-function-derivative weights.
    Modes are sorted as 1, xi, v^2, v, v^4, v^3, ..., interleaved in parity,
    where xi = (E - Ef)/T is the dimensionless radial coordinate, and orthonormalized
    with a Gauss-Legendre quadrature. Note that n_r = 1 samples just the Fermi circle.
    """

    n_r: int  #: Number of radial nodes and modes
    xi: torch.Tensor  #: Radial quadrature nodes on (E - Ef)/T space
    w_xi: torch.Tensor  #: Radial integration weights
    from_modes: torch.Tensor  #: Transform to modes (modes in columns)
    to_modes: torch.Tensor  #: Transform to modes (modes in rows)

    def __init__(
        self,
        n_r: int,
        T: float = 1.0,
        xi_max: float = 6.0,
    ) -> None:
        self.n_r = n_r
        self.T = T
        self.xi_max = xi_max
        if n_r == 1:
            self.xi = torch.zeros(1, device=rc.device)
            self.w_xi = torch.ones(1, device=rc.device)
            self.from_modes = torch.ones((1, 1), device=rc.device)
            self.to_modes = torch.ones((1, 1), device=rc.device)
            return

        # Collocation: Gauss-Legendre in u = tanh(xi/2)/u_max
        x_std, w_std = np.polynomial.legendre.leggauss(n_r)
        u_max = np.tanh(0.5 * xi_max)
        xi = 2.0 * np.arctanh(u_max * x_std)
        w_xi = (u_max / (2.0 * T)) * w_std

        def get_basis(xi: np.ndarray) -> np.ndarray:
            """Un-normalized basis functions: [1, x/xi_max, u^2, u, u^4, u^3, ...]"""
            u = np.tanh(0.5 * xi) / u_max
            cols = [np.ones_like(xi), xi / xi_max]
            pe, po = 2, 1
            while len(cols) < n_r:
                cols.append(u**pe)
                pe += 2
                if len(cols) < n_r:
                    cols.append(u**po)
                    po += 2
            return np.stack(cols[:n_r], axis=-1)

        # Orthonormalize against a fine bare-xi Gauss quadrature:
        xf, wf = np.polynomial.legendre.leggauss(128)
        xf = xi_max * xf
        wf = (xi_max * wf) * ((1.0 / (4.0 * T)) / np.cosh(0.5 * xf) ** 2)
        A = get_basis(xf) * np.sqrt(wf)[:, None]
        _Q, R = np.linalg.qr(A)
        sign = np.sign(np.diag(R))
        sign[sign == 0] = 1.0
        R = sign[:, None] * R
        C = np.linalg.solve(R, np.eye(n_r))  # psi = feats(.) @ C
        transform = get_basis(xi) @ C  # psi at the nodes
        inv_transform = np.linalg.inv(transform)
        metric = (A @ C).T @ (A @ C)
        err = max(
            float(np.max(np.abs(inv_transform @ transform - np.eye(n_r)))),
            float(np.max(np.abs(metric - np.eye(n_r)))),
        )
        if err > 1e-8:
            raise RuntimeError(f"RadialBasis orthonormality failed (err={err:.2e})")
        self.xi = rc.from_np(xi)
        self.w_xi = rc.from_np(w_xi)
        self.from_modes = rc.from_np(transform)
        self.to_modes = rc.from_np(inv_transform)


def wrap_dense_reflector(
    reflector: Callable[[torch.Tensor], torch.Tensor], n: torch.Tensor, Nk: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Wrap linear reflection operator with a dense matrix (if memory allows)."""
    n_wall = len(n)
    refl_budget = float(os.environ.get("QIMPY_REFL_BUDGET_GB", "4.0")) * 2**30
    if n_wall * (Nk**2) * 8 <= refl_budget:
        eye = torch.eye(Nk, device=rc.device, dtype=n.dtype)
        matrix = torch.empty(n_wall, Nk, Nk, device=rc.device, dtype=n.dtype)
        for i_basis, basis in enumerate(eye):
            matrix[..., i_basis] = reflector(basis[None, None, :].expand(1, n_wall, Nk))
        log.info(f"Wall reflector: dense matrix ({n_wall} edges x {Nk}^2)")
        return DenseReflector(matrix)
    else:
        log.info("Wall reflector: apply per step (dense matrix too large)")
        return reflector


@dataclass
class DenseReflector:
    matrix: torch.Tensor

    def __call__(self, rho: torch.Tensor) -> torch.Tensor:
        return torch.einsum("eic, ec -> ei", self.matrix, rho)
