"""k-space representations (numerical backends) for the FermiSurface model.

The physical model (`FermiSurface`) owns the collision physics; a representation
owns the discrete k-grid and the map between stored delta-f and the modal e-e
basis, plus the matching boundary operators and observables.  The model applies
the collision through ``representation.apply_collision(rho, modal_op)``, where
``modal_op`` is the model's representation-agnostic modal operator
``a -> -rates*a + ee.a_dot(a) + k_speed*G(a)``.

Two backends:
  * DeltaK    -- delta-k collocation on (radial x angular) nodes near the Fermi
                 surface, |v| = vF, lab-centred tensor-product transform.  The
                 Fermi-circle / small-drift regime (Nr=1 = pure Fermi circle).
  * Cartesian -- a fixed uniform 2D k-grid, v = k/m*, with a drift-centred
                 per-cell projection onto the SAME modal basis.  The large-drift
                 full-f regime.  (see _cartesian.py)
"""

from __future__ import annotations
from typing import Callable
import numpy as np
import torch

from qimpy import rc, TreeNode
from qimpy.mpi import ProcessGrid
from qimpy.io import CheckpointPath


class KRepresentation(TreeNode):
    """Numerical k-space backend for `FermiSurface`.

    Subclasses build the grid in ``__init__`` and set ``Nk``, ``wk``, ``v``
    (transport velocity, shape ``(Nk, 2)``), then provide ``apply_collision``,
    the boundary operators, and the observables.
    """

    Nk: int
    wk: torch.Tensor  # (Nk,) Brillouin-zone/mode integration weights
    v: torch.Tensor  # (Nk, 2) transport velocity
    constraint_factors: tuple[torch.Tensor, torch.Tensor] | None

    def apply_collision(self, rho: torch.Tensor, modal_op: Callable) -> torch.Tensor:
        raise NotImplementedError

    def get_contactor(self, n: torch.Tensor, **kwargs) -> Callable:
        raise NotImplementedError

    def get_reflector(self, n: torch.Tensor) -> Callable:
        raise NotImplementedError

    # ---- observables: scalars on cells, vector moments as face fluxes ----
    def get_observables(self) -> torch.Tensor:  # (Nk,) per-k density measure
        raise NotImplementedError

    def get_observable_names(self) -> list[str]:
        raise NotImplementedError

    def measure_observables(self, rho: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_flux_indices(self) -> list[int]:
        raise NotImplementedError


class DeltaK(KRepresentation):
    """delta-k collocation on (k_r, theta_q); |v| = vF; lab-centred modal transform.

    Storage is ``Nr * N_theta`` collocation points; the modal transform is the
    tensor product of the angular Fourier and radial polynomial bases (the model's
    ``angular``/``radial``).  ``Nr=1`` is the pure Fermi-circle limit.
    """

    def __init__(
        self,
        *,
        fermi_surface,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        super().__init__()
        fs = fermi_surface
        self.fs = fs
        self.Nk = fs.n_r * fs.n_theta
        self.wk = self._density_weight()
        # per-collocation velocity: |v| = vF for every radial point (shell linearization)
        theta_q = fs.angular.theta
        v_per_theta = fs.vF * torch.stack(
            [torch.cos(theta_q), torch.sin(theta_q)], dim=-1
        )
        self.v = v_per_theta.repeat(fs.n_r, 1)  # (Nr*N_theta, 2)

        # Initialize constraint operator from mode transformations (if needed):
        self.constraint_factors = None
        eye = torch.eye(self.Nk, device=rc.device, dtype=self.v.dtype)
        proj = self.from_modes(self.to_modes(eye))  # (Nk, Nk) projector
        ghost = eye - proj  # onto unrepresented DOFs
        if ghost.abs().max() > 1e-10:  # only if oversampled
            U, S, Vh = torch.linalg.svd(ghost)
            tol = float(S[0]) * torch.finfo(self.v.dtype).eps ** 0.5
            r = int((S > tol).sum())
            self.constraint_factors = (
                (U[:, :r] * S[:r]).contiguous(),
                Vh[:r].T.contiguous(),
            )

    # ---- modal transforms (tensor product of angular and radial) ----
    def to_modes(self, f: torch.Tensor) -> torch.Tensor:
        """Nodal (..., Nr*N_theta) -> modal (..., Nr*(2M+1))."""
        fs = self.fs
        f4 = f.unflatten(-1, (fs.n_r, fs.n_theta))
        a_t = torch.einsum("cq,...rq->...rc", fs.angular.to_modes, f4)
        a = torch.einsum("nr,...rc->...nc", fs.radial.to_modes, a_t)
        return a.flatten(-2, -1)

    def from_modes(self, a: torch.Tensor) -> torch.Tensor:
        """Modal (..., Nr*(2M+1)) -> nodal (..., Nr*N_theta)."""
        fs = self.fs
        a4 = a.unflatten(-1, (fs.n_r, fs.n_theta))
        f_r = torch.einsum("rn,...nc->...rc", fs.radial.from_modes, a4)
        f = torch.einsum("qc,...rc->...rq", fs.angular.from_modes, f_r)
        return f.flatten(-2, -1)

    def apply_collision(self, rho: torch.Tensor, modal_op: Callable) -> torch.Tensor:
        return self.from_modes(modal_op(self.to_modes(rho)))

    # ---- observables: scalar fields on cells; vector moments as face fluxes ----
    def _density_weight(self) -> torch.Tensor:
        fs = self.fs
        w_r = fs.radial.w_xi / torch.sqrt(fs.radial.w_xi.sum())
        w_theta = torch.full_like(fs.angular.theta, 1.0 / fs.n_theta)
        return (w_r[:, None] * w_theta[None, :]).flatten()

    def _eps_node(self) -> torch.Tensor:
        fs = self.fs  # per-node energy
        return (fs.mu + fs.radial.xi * fs.T).repeat_interleave(fs.n_theta)

    def get_observable_names(self) -> list[str]:
        return ["n", "vx", "vy", "e"]

    def get_observables(self) -> torch.Tensor:
        ones = torch.ones((1, self.Nk), device=rc.device)
        return torch.cat((ones, self.v.T, self._eps_node()[None]), dim=0)

    def get_flux_indices(self) -> list[int]:
        return [0]

    def get_contactor(self, n: torch.Tensor, **kwargs) -> Callable:
        return _DeltaKContactor(self.fs, n, **kwargs)

    def get_reflector(self, n: torch.Tensor) -> Callable:
        return _DeltaKReflector(self.fs, n, self.fs.specularity)


class _DeltaKContactor:
    """Reservoir ghost for the delta-k representation.

    The reservoir is a drifted Fermi-Dirac at (mu + dmu, T) with PHYSICAL inward
    drift velocity ``vD`` (along -n, into the device), i.e. a shell shift
    s(theta) = (dmu - kF vD cos(theta-phi) - m* vD^2/2) / T.

    nonlinear=False (default): the EXACT LINEARIZATION -- nodal storage
        Phi = dmu - kF vD cos(theta-phi), built in modes (m=0 and m=1 only).
    nonlinear=True: the exact Pauli-bounded deviation of the shifted FD,
        Phi(xi, theta) = 2T (1 + cosh xi) [expit(s - xi) - expit(-xi)]
        (at Nr=1 this is the saturating 2T tanh(s/2) contact), band-limited once
        through the representation's modal transform.  Its small-s limit is
        exactly the linear ghost, so the flag changes fidelity, not convention.
    """

    def __init__(
        self,
        fs,
        n: torch.Tensor,
        *,
        dmu: float = 0.0,
        vD: float = 0.0,
        nonlinear: bool = False,
    ) -> None:
        n = n.to(rc.device)
        Nsel = n.shape[0]
        phi = torch.atan2(n[:, 1], n[:, 0])
        rep = fs.representation
        if not nonlinear:
            cm = torch.zeros(
                (Nsel, fs.n_r, fs.n_theta), device=rc.device, dtype=n.dtype
            )
            m_max = fs.angular.m_max
            cm[:, 0, 0] = dmu
            cm[:, 0, 1] = -(fs.kF * vD) * torch.cos(phi)
            cm[:, 0, m_max + 1] = -(fs.kF * vD) * torch.sin(phi)
            self.rho_contact = rep.from_modes(cm.flatten(-2, -1))
        else:
            theta = fs.angular.theta  # (N_theta,)
            xi = fs.radial.xi  # (Nr,)
            m_star = fs.kF / fs.vF
            s = (
                dmu
                - fs.kF * vD * torch.cos(theta[None, :] - phi[:, None])
                - 0.5 * m_star * vD * vD
            ) / fs.T  # (Nsel, N_theta)
            xi_b = xi[None, :, None]  # (1, Nr, 1)
            s_b = s[:, None, :]  # (Nsel, 1, N_theta)
            dfe = torch.special.expit(s_b - xi_b) - torch.special.expit(-xi_b)
            Phi = 2.0 * fs.T * (1.0 + torch.cosh(xi_b)) * dfe
            Phi = Phi.reshape(Nsel, fs.n_r * fs.n_theta)
            self.rho_contact = rep.from_modes(rep.to_modes(Phi))  # band-limit once

    def __call__(self, t: float) -> torch.Tensor:
        return self.rho_contact


class _DeltaKReflector:
    """Specular block-rotation per harmonic (fraction s) + diffuse (1-s) refill,
    with mass and tangential-momentum conservation, all in modes."""

    def __init__(self, fs, n: torch.Tensor, specularity: float) -> None:
        n = n.to(rc.device)
        self.fs = fs
        self.s = float(specularity)
        self.phi = torch.atan2(n[:, 1], n[:, 0])
        self.angle = 2.0 * self.phi + np.pi
        theta = fs.angular.theta
        v_dot_n = fs.vF * (
            n[:, 0:1] * torch.cos(theta)[None, :]
            + n[:, 1:2] * torch.sin(theta)[None, :]
        )
        self.adn_pos = v_dot_n.clamp(min=0.0)
        self.adn_neg = v_dot_n.clamp(max=0.0)
        self.w_in = (-self.adn_neg).sum(-1).clamp(min=1e-300)
        self.T_to_rad_0 = fs.radial.to_modes[0, :]
        self.psi_0 = fs.radial.from_modes[:, 0]

    def _specular_modal(self, a_modal: torch.Tensor) -> torch.Tensor:
        # Per-harmonic block rotation, vectorized over all m at once (a Python
        # loop over m is launch-bound and dominates the per-step cost at large
        # M_theta): out_c + i*out_s = e^{i m angle} (a_c - i a_s) for m = 1..M.
        a = a_modal.unflatten(-1, (self.fs.n_r, self.fs.n_theta))
        out = a.clone()
        m_max = self.fs.angular.m_max
        m = torch.arange(1, m_max, device=a_modal.device, dtype=a_modal.dtype)
        ang = self.angle[:, None] * m  # (Nsel, M)
        c = torch.cos(ang).unsqueeze(-2)  # (Nsel, 1, M) -> bcast Nr
        s = torch.sin(ang).unsqueeze(-2)
        a_c = a[..., 1:m_max]  # (..., Nsel, Nr, M-1) cos coeffs
        a_s = -a[..., (m_max + 1) :]  # (..., Nsel, Nr, M-1) -sin coeffs
        out[..., 1:m_max] = c * a_c - s * a_s
        out[..., (m_max + 1) :] = s * a_c + c * a_s
        out[..., m_max] *= torch.cos(self.angle * m_max)[:, None]
        return out.reshape(a_modal.shape)

    def __call__(self, uM_dk: torch.Tensor) -> torch.Tensor:
        fs = self.fs
        rep = fs.representation
        uM_modal = rep.to_modes(uM_dk)
        spec_dk = rep.from_modes(self._specular_modal(uM_modal))
        s = uM_dk.shape
        uM4 = uM_dk.unflatten(-1, (self.fs.n_r, self.fs.n_theta))
        sp4 = spec_dk.unflatten(-1, (self.fs.n_r, self.fs.n_theta))
        T_to_r = fs.radial.to_modes
        T_from_r = fs.radial.from_modes
        uM_n_q = torch.einsum("nr,...arq->...anq", T_to_r, uM4)
        sp_n_q = torch.einsum("nr,...arq->...anq", T_to_r, sp4)
        sin_q = (
            torch.cos(self.phi)[:, None] * torch.sin(fs.angular.theta)[None, :]
            - torch.sin(self.phi)[:, None] * torch.cos(fs.angular.theta)[None, :]
        )
        beta = (self.adn_neg * sin_q).sum(-1)
        gamma = (self.adn_neg * sin_q**2).sum(-1)
        adn_pos = self.adn_pos[:, None, :]
        adn_neg = self.adn_neg[:, None, :]
        sin_b = sin_q[:, None, :]
        vF = fs.vF
        F_out_mass_n = (adn_pos * uM_n_q).sum(-1)
        F_in_spec_mass_n = (adn_neg * sp_n_q).sum(-1)
        F_out_tang_n = vF * (adn_pos * sin_b * uM_n_q).sum(-1)
        F_in_spec_tang_n = vF * (adn_neg * sin_b * sp_n_q).sum(-1)
        F_M_n = F_out_mass_n + self.s * F_in_spec_mass_n
        F_T_n = F_out_tang_n + F_in_spec_tang_n
        det = -vF * (self.w_in * gamma + beta * beta)
        det_b = det[:, None]
        b1 = -F_M_n
        b2 = -self.s * F_T_n
        D = (b1 * (vF * gamma)[:, None] - b2 * beta[:, None]) / det_b
        T = ((-self.w_in)[:, None] * b2 - (vF * beta)[:, None] * b1) / det_b
        D_r = torch.einsum("rn,...an->...ar", T_from_r, D)
        T_r = torch.einsum("rn,...an->...ar", T_from_r, T)
        u_added = D_r.unsqueeze(-1) + T_r.unsqueeze(-1) * sin_q.unsqueeze(-2)
        return self.s * spec_dk + u_added.reshape(s)
