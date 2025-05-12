from __future__ import annotations
from typing import Callable
from functools import cache

import numpy as np
import torch

from qimpy import rc, MPI
from qimpy.mpi import ProcessGrid, BufferView, TaskDivision
from qimpy.profiler import stopwatch
from qimpy.io import CheckpointPath
from . import Material

class SingleBandModel(Material):
    """Fermi-circle representation suitable for graphene and 2DEGs."""

    kF: float  #: Fermi wave-vector
    vF: float  #: Fermi velocity
    tau_inv_p: float  #: Momentum relaxation rate
    tau_inv_ee: float  #: Electron internal scattering rate (momentum-conserving)

    def __init__(
        self,
        *,
        kF: float,
        vF: float,
        delta_k: float,
        mass: float,
        T: float,
        N_theta: int,
        N_r: int,
        tau_p: float,
        tau_ee: float,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Initialize ab initio material.

        Parameters
        ----------
        kF
            :yaml:`Fermi wave vector in atomic units.`
        vF
            :yaml:`Fermi velocity in atomic units.`
        N_theta
            :yaml:`Number of k along Fermi circle.`
        """
        self.kF = kF
        self.vF = vF
        self.delta_k = delta_k
        self.mass = mass
        self.T = T
        self.tau_inv_p = 1.0 / tau_p
        self.tau_inv_ee = 1.0 / tau_ee
        self.N_theta = N_theta
        self.N_r = N_r
        super().__init__(
            wk=1.0 / N_theta, # TODO: put in proper integration weight
            nk=N_theta*N_r,
            n_bands=1,
            n_dim=2,
            checkpoint_in=checkpoint_in,
            process_grid=process_grid,
        )

        # Create cylindrical grid on the Fermi surface:
        dk_theta = 2 * np.pi / N_theta # TODO: N_theta -> N_k_theta?
        k_theta = (0.5 + torch.arange(N_theta, device=rc.device)) * dk_theta
        self.dk_theta = dk_theta

        dk_r = 2 * delta_k / N_r # Domain extent is [kF - delta_k, kF + delta_k]
        k_r = kF - delta_k + (0.5 + torch.arange(N_r, device=rc.device)) * dk_r
        self.dk_r = dk_r

        k_hat = torch.stack([k_theta.cos(), k_theta.sin()], dim=-1) # N_theta x 2

        k = torch.einsum("r, t i -> r t i", k_r, k_hat)
        k = k.flatten(0, 1)

        self.k[:] = k[self.k_mine]
        self.v[:, 0, :] = (self.k[:]/self.kF) * self.vF
        self.v_all = torch.zeros_like(self.v)
        self.v_all[:, 0, :] = (k/self.kF) * self.vF
        self.E[:, 0] = (torch.linalg.norm(self.k, dim=-1) - self.kF)*self.vF
        self.rho_init = torch.special.expit(-self.E[:, 0]/self.T) # Don't use self.rho0 since it is already defined in material

        # ee scattering
        N_k = N_r*N_theta
        k_0 = torch.arange(N_k, device=rc.device)[:, None, None, None]
        k_1 = torch.arange(N_k, device=rc.device)[None, :, None, None]
        k_2 = torch.arange(N_k, device=rc.device)[None, None, :, None]
        k_3 = torch.arange(N_k, device=rc.device)[None, None, None, :]
        self.k_0 = k_0; self.k_1 = k_1; self.k_2 = k_2; self.k_3 = k_3

        energy_eqn = self.E[k_0, 0] + self.E[k_3, 0] - self.E[k_1, 0] - self.E[k_2, 0]
        momentum_eqns = self.k[k_0] + self.k[k_3] - self.k[k_1] - self.k[k_2]

        threshold =   1e-10
        self.nonzero_indices = (
                                (torch.abs(momentum_eqns[..., 0]) < threshold)
                                * (torch.abs(momentum_eqns[..., 1]) < threshold) )
        print(f"{self.nonzero_indices.sum() = }")

        q_1 = torch.sum(self.k[k_1] - self.k[k_0], dim=-1)
        q_2 = torch.sum(self.k[k_2] - self.k[k_0], dim=-1)
        kappa = 1
        k1_k0 = 1 / (q_1 + kappa + 1e-15)
        k2_k0 = 1 / (q_2 + kappa + 1e-15)
        M_ee_sqr = (k1_k0 ** 2 + k2_k0 ** 2 + (k1_k0 - k2_k0) ** 2)

        rho0 = self.rho_init
        a0 = rho0[..., k_3] * (1 - rho0[..., k_1] - rho0[..., k_2]) \
             + rho0[..., k_1] * rho0[..., k_2]

        a3 = rho0[..., k_0] * (1 - rho0[..., k_1] - rho0[..., k_2]) \
             + rho0[..., k_1] * rho0[..., k_2]

        a1 = rho0[..., k_2] * (-1 + rho0[..., k_0] + rho0[..., k_3]) \
             - rho0[..., k_0] * rho0[..., k_3]

        a2 = rho0[..., k_1] * (-1 + rho0[..., k_0] + rho0[..., k_3]) \
             - rho0[..., k_0] * rho0[..., k_3]

        a01 = rho0[..., k_2] - rho0[..., k_3]
        a02 = rho0[..., k_1] - rho0[..., k_3]
        a03 =  1 - rho0[..., k_1] - rho0[..., k_2]
        a12 = -1 + rho0[..., k_0] + rho0[..., k_3]
        a13 = rho0[..., k_2] - rho0[..., k_0]
        a23 = rho0[..., k_1] - rho0[..., k_0]

        a012 = 1; a013 = -1; a023 = -1; a123 = 1

        self.a0 = M_ee_sqr * a0
        self.a1 = M_ee_sqr * a1
        self.a2 = M_ee_sqr * a2
        self.a3 = M_ee_sqr * a3

        self.a01 = M_ee_sqr * a01
        self.a02 = M_ee_sqr * a02
        self.a03 = M_ee_sqr * a03
        self.a12 = M_ee_sqr * a12
        self.a13 = M_ee_sqr * a13
        self.a23 = M_ee_sqr * a23

        self.a012 = M_ee_sqr * a012
        self.a013 = M_ee_sqr * a013
        self.a023 = M_ee_sqr * a023
        self.a123 = M_ee_sqr * a123

    def get_contactor(
        self, n: torch.Tensor, *, dmu: float = 0.0, vD: float = 0.0
    ) -> Callable[[float], torch.Tensor]:
        """Return contact distribution function for specified chemical potential
        shift and drift velocity. Note that positive vD corresponds to current
        flowing into the device (along -n), while negative vD flows out (along +n)."""
        vel_drift = vD * n
        rho_contact = torch.special.expit(-(self.E[:, 0] - vel_drift @ self.k.T - dmu)/self.T) - self.rho_init
        return lambda t: rho_contact  # TODO: add time-dependence options

    def get_reflector(self, n: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        return SpecularReflector(n, self.v_all, self.N_r, self.comm, self.k_division)

    @stopwatch
    def rho_dot(self, rho: torch.Tensor, t: float) -> torch.Tensor:
        # return torch.zeros_like(rho)  # no scattering

        drho = rho - self.rho_init
        k_0 = self.k_0; k_1 = self.k_1; k_2 = self.k_2; k_3 = self.k_3

        integrand1 =  self.a0 * drho[..., k_0] + self.a1 * drho[..., k_1] \
                    + self.a2 * drho[..., k_2] + self.a3 * drho[..., k_3]

        integrand2 =  self.a01 * drho[..., k_0] * drho[..., k_1] \
                    + self.a02 * drho[..., k_0] * drho[..., k_2] \
                    + self.a03 * drho[..., k_0] * drho[..., k_3] \
                    + self.a12 * drho[..., k_1] * drho[..., k_2] \
                    + self.a13 * drho[..., k_1] * drho[..., k_3] \
                    + self.a23 * drho[..., k_2] * drho[..., k_3]

        integrand3 =  self.a012 * drho[..., k_0] * drho[..., k_1] * drho[..., k_2] \
                    + self.a013 * drho[..., k_0] * drho[..., k_1] * drho[..., k_3] \
                    + self.a023 * drho[..., k_0] * drho[..., k_2] * drho[..., k_3] \
                    + self.a123 * drho[..., k_1] * drho[..., k_2] * drho[..., k_3]

        integrand_expanded = -(integrand1 + integrand2 + integrand3)

        result = torch.einsum("...abcd -> ...a", integrand_expanded * self.nonzero_indices)

        print(f"{integrand_expanded.shape = }")
        print(f"{result.shape = }")

        exit(0)

    def get_observable_names(self) -> list[str]:
        return ["q"]  # charge

    @cache
    def get_observables(self, t: float) -> torch.Tensor:
        Nkbb = len(self.k)  # since n_bands = 1
        return torch.ones((1, Nkbb), device=rc.device)  # charge observable

def dirac_delta(x: torch.tensor) -> torch.Tensor:
    eps = 1e-15
    return torch.exp(-x**2/(2*eps))/(2*np.pi*eps)**0.5

class SpecularReflector:
    """Reflect velocities specularly i.e. with reflection angle = incidence angle."""

    def __init__(
        self, n: torch.Tensor, v: torch.Tensor, N_energy: int, comm: MPI.Comm, k_division: TaskDivision
    ) -> None:
        assert v.shape[1] == 1  # only for single-band case
        # Find which theta reflects into the first one:
        v0_normal = torch.einsum("ri, rj, j -> ri", n, n, v[0, 0])
        v0_reflected = v[None, 0, 0] - 2 * v0_normal
        v0_diff = (v0_reflected[:, None] - v[None, :, 0]).norm(dim=-1)
        i0_reflected = v0_diff.argmin(dim=1)
        comm.Bcast(BufferView(i0_reflected))  # ensure indices consistent
        # Map the rest correspondingly:
        nk = k_division.n_tot
        nkr = N_energy
        nkt = nk//nkr
        i_reflected = (i0_reflected[:, None] - torch.arange(nkt, device=rc.device)) % nkt

        # Vectorized code below corresponds to the following loop:
        # i_reflected_flat = torch.zeros(nr*nkr*nkt, device=rc.device, dtype=int)
        # for ir/ikr/ikt in range(nr/nkr/nkt):
        #     i_reflected_flat[ikt + nkt*(ikr + nkr*ir)] = nkt*(ikr + nkr*ir) + i_reflected[ir, ikt]
        nr = len(n)
        ir = torch.arange(nr, device=rc.device)[:, None, None]
        ikr = torch.arange(nkr, device=rc.device)[None, :, None]
        self.i_reflected_flat = (nkt*(ikr + nkr*ir) + i_reflected[:, None, :]).flatten()

    def __call__(self, rho: torch.Tensor) -> torch.Tensor:
        rho_flat = rho.flatten(1, 2)  # flatten r and k indices
        out_flat = rho_flat[:, self.i_reflected_flat]
        return out_flat.unflatten(1, rho.shape[1:])  # restore r and k
