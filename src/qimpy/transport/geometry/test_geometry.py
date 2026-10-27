from .. import Transport

import torch
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from qimpy import rc
from qimpy.io import log_config


def test_geometry():

    transport = Transport(
        fermi_surface=dict(kF=1.5, vF=0.8, M_theta=15),
        geometry=dict(
            contacts={}, mesh=dict(rect=dict(Lx=1.1, Ly=0.9, Nx=33, Ny=27, contacts={}))
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
    geometry.rho[: cell_div.n_mine, 0] = torch.exp(
        -0.5 * ((r - r0) / sigma).square().sum(dim=1)
    )
    print(transport.material.transport_velocity)

    def check_results(i_step: int, t: float) -> None:
        """Intercept Geometry.update_probes to check results"""
        x, y = rc.to_np(r).T
        norm = SymLogNorm(vmin=0, vmax=1, linthresh=0.1)
        """
        for ik, rho_k in enumerate(geometry.rho.T):
            plt.figure()
            plt.scatter(x, y, c=rc.to_np(rho_k), vmin=0, vmax=1, cmap="Blues")
            plt.savefig(
                f"/home/shankar/Downloads/tmp/{ik}-{i_step}.png", bbox_inches="tight"
            )
            plt.close()
        """
        plt.figure()
        plt.scatter(x, y, c=rc.to_np(geometry.rho.sum(dim=-1)), norm=norm, cmap="Blues")
        plt.savefig(
            f"/home/shankar/Downloads/tmp/mean-{i_step}.png", bbox_inches="tight"
        )
        plt.close()

    transport.geometry.update_probes = check_results
    transport.run()
    print(f"{transport.geometry.rho.shape}")


def main():
    log_config()
    rc.init()
    test_geometry()


if __name__ == "__main__":
    main()
