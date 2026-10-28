import pytest
import torch

from qimpy import rc
from qimpy.io import log_config
from .. import Transport


@pytest.mark.mpi_skip
def test_geometry():
    Lx = 1.1
    Ly = 0.9
    vF = 0.8
    transport = Transport(
        fermi_surface=dict(kF=1.5, vF=vF, n_theta=8),
        geometry=dict(
            contacts={}, mesh=dict(rect=dict(Lx=Lx, Ly=Ly, Nx=44, Ny=36, contacts={}))
        ),
        time_evolution=dict(t_max=2.0, dt_probe=0.1, dt_save=10.0),
    )

    # Create test distribution:
    geometry = transport.geometry
    mesh = geometry.mesh
    cell_div = mesh.cell_division
    r = rc.from_np(
        mesh.vertices[mesh.cells[cell_div.i_start : cell_div.i_stop, :-1]].mean(axis=1)
    )
    r0 = torch.tensor([0.5, 0.4], device=rc.device)
    sigma = 0.1
    v = transport.material.transport_velocity
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
        rel_err = ((n - n_pred).abs().sum() / n_pred.abs().sum()).item()
        rel_err_rate = rel_err / (i_step + 1)
        print(f"{rel_err_rate=:.2g}")
        assert rel_err_rate < 2e-4

    transport.geometry.update_probes = check_results
    transport.run()


def main():
    log_config()
    rc.init()
    test_geometry()


if __name__ == "__main__":
    main()
