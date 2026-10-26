from __future__ import annotations

import torch
import pytest

from qimpy import rc
from qimpy.mpi import ProcessGrid

pytestmark = pytest.mark.mpi_skip  # All tests in this module are serial

KF = 7.5e-3
M_STAR = 0.067
T0 = 1.3301e-5
EPS_B = 12.9
KAPPA = 0.0

ee_scattering = dict(epsilon_bg=EPS_B, n_xi=8, n_phi=64, n_xi_proj=6)


def _fs(**kw):
    from qimpy.transport.material import FermiSurface

    return FermiSurface(
        kF=KF,
        vF=KF / M_STAR,
        M_theta=3,
        Nr=1,
        T=T0,
        tau_p=float("inf"),
        process_grid=ProcessGrid("rk", (1, 1)),
        **kw,
    )


def test_ee_conservation() -> None:
    torch.set_default_dtype(torch.float64)
    fs = _fs(ee_scattering=dict(**ee_scattering, nonlinear=True))
    dim = 2 * fs.M_theta + 1
    torch.manual_seed(1)
    a = 1e-2 * torch.randn(4, fs.Nr * dim, dtype=torch.float64, device=rc.device)
    out = fs.ee_scattering.a_dot(a).reshape(4, fs.Nr, dim)
    scale = float(out.abs().max().clamp(min=1e-300))
    # m = 0 is the particle-number channel; m = 1 (cos, sin) is momentum.
    assert float(out[:, :, 0].abs().max()) / scale < 1e-9, "particle leak"
    assert float(out[:, :, 1:3].abs().max()) / scale < 1e-9, "momentum leak"


def test_ee_linearity() -> None:
    torch.set_default_dtype(torch.float64)
    dim = 2 * 3 + 1
    torch.manual_seed(2)
    a = 1e-2 * torch.randn(1, dim, dtype=torch.float64, device=rc.device)

    lin = _fs(ee_scattering=dict(**ee_scattering, nonlinear=False))
    o1 = lin.ee_scattering.a_dot(a)
    o2 = lin.ee_scattering.a_dot(2.0 * a)
    err = float((o2 - 2.0 * o1).abs().max() / o1.abs().max().clamp(min=1e-300))
    assert err < 1e-12, f"linear operator is not linear: {err:.2e}"
