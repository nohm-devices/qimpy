import numpy as np
import pytest
import torch

from qimpy import rc, log
from qimpy.io import log_config
from qimpy.mpi import globalreduce
from .. import Transport
from . import Mesh


@pytest.mark.parametrize(
    "h, n_theta, rotation_deg, rel_err_rate_bound",
    (
        [
            (0.025, 8, 0.0, 2.0e-2),  # baseline
            (0.018, 8, 0.0, 1.0e-2),  # quadratic improvement with h
            (0.025, 16, 0.0, 2.0e-2),  # no effect of n_theta when aligned
            (0.025, 8, 17.9, 4.5e-1),  # reflections worsen when not aligned,
            (0.025, 32, 17.9, 1.0e-1),  # but improve with n_theta
        ]
        if rc.n_procs == 1
        else [(0.025, 8, 0.0, 2.0e-2)]
    ),  # just the baseline with MPI
)
def test_gaussian_pong(
    h: float, n_theta: int, rotation_deg: float, rel_err_rate_bound: float
) -> None:
    """Bounce a k-weighted gaussian blob with net charge and drift velocity
    within a rectangular domain, with and without rotations, and compare the
    full transient to an analytical (forward ray-trace) solution.
    This test confirms quadratic error scaling with grid resolution, as well as
    linear error scaling with angular resolution for the non-alignd (rotated) case."""
    Lx = 1.1
    Ly = 0.9
    vF = 0.8

    # Construct mesh with vertex rotation:
    Nx = int(np.round(Lx / h))
    Ny = int(np.round(Ly / h))
    mesh = Mesh(rect=dict(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, contacts={}))
    cos = np.cos(np.deg2rad(rotation_deg))
    sin = np.sin(np.deg2rad(rotation_deg))
    rot = np.array([[cos, -sin], [sin, cos]])
    vertices = np.copy(mesh.vertices)  # un-rotated copy
    mesh.vertices[:] = vertices @ rot  # mesh has rotated version

    transport = Transport(
        fermi_surface=dict(kF=1.5, vF=vF, n_theta=n_theta),
        geometry=dict(contacts={}, mesh=mesh),
        time_evolution=dict(t_max=2.01, dt_probe=0.5, dt_save=10.0),
    )

    # Create test distribution in unrotated frame:
    geometry = transport.geometry
    mesh = geometry.mesh
    cell_div = mesh.cell_division
    cells_mine = mesh.cells[cell_div.i_start : cell_div.i_stop, :-1]
    r = rc.from_np(vertices[cells_mine].mean(axis=1))  # unrotated frame
    r0 = torch.tensor([0.5, 0.4], device=rc.device)
    sigma = 0.1
    v = transport.material.transport_velocity @ rc.from_np(rot.T)  # unrotated frame
    wk = (v @ torch.tensor([0.8, 0.6], device=rc.device)) / vF + 0.5  # charge + drift
    geometry.rho[: cell_div.n_mine] = wk * torch.exp(
        -0.5 * ((r - r0) / sigma).square().sum(dim=1, keepdim=True)
    )
    L = torch.tensor([Lx, Ly], device=rc.device)

    # Pre-calculate nearest boundary reflections of r for rho prediction below:
    x, y = r.T
    x_reflected = torch.where(x < 0.5 * Lx, -x, 2 * Lx - x)
    y_reflected = torch.where(y < 0.5 * Ly, -y, 2 * Ly - y)
    r_reflected = torch.stack(
        (
            torch.stack((x, y), dim=-1),
            torch.stack((x, y_reflected), dim=-1),
            torch.stack((x_reflected, y), dim=-1),
            torch.stack((x_reflected, y_reflected), dim=-1),
        )
    )

    def check_results(i_step: int, t: float) -> None:
        """Intercept Geometry.update_probes to check results"""
        if not i_step:
            return  # don't test the initial distribution
        # Predict where the blob should be at each k:
        frac = (r0 + v * t) / L
        i_cell = frac.floor().to(torch.int)
        frac -= i_cell
        frac_reflected = torch.where(i_cell % 2 == 0, frac, 1 - frac)
        r_t = (frac_reflected * L)[:, None, None]
        # Compute corresponding density and error:
        kernel = torch.exp(-0.5 * ((r_reflected - r_t) / sigma).square().sum(dim=-1))
        n_pred = wk @ kernel.sum(dim=1)
        n = geometry.rho.sum(dim=-1)
        rel_err = (
            globalreduce.sum((n - n_pred).abs(), group=geometry.group)
            / globalreduce.sum(n_pred.abs(), group=geometry.group)
        ).item()
        rel_err_rate = rel_err / t
        log.info(f"{rel_err_rate=:.2g}")
        assert rel_err_rate < rel_err_rate_bound

    transport.geometry.update_probes = check_results
    transport.run()


def main():
    log_config()
    pytest.main([__file__, "-vs"])


if __name__ == "__main__":
    main()
