from __future__ import annotations

import numpy as np
import torch

from qimpy import TreeNode, rc
from qimpy.io import CheckpointPath
from qimpy.transport.advect import Advect
from . import BicubicPatch, parse_svg


class Geometry(TreeNode):
    """Geometry specification."""

    vertices: torch.Tensor
    edges: torch.Tensor
    quads: torch.Tensor
    adjacency: torch.Tensor

    # v_F and N_theta should eventually be material paramteters
    def __init__(
        self,
        *,
        svg_file: str,
        N: tuple[int, int],
        N_theta: int,
        v_F: torch.Tensor,
        # For now, we are testing horizontal or diagonal advection
        diag: bool,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Initialize geometry parameters.

        Parameters
        ----------
        svg_file
            :yaml:`Path to an SVG file containing the input geometry.
        """
        super().__init__()
        self.patch_set = parse_svg(svg_file)
        self.patches = []
        # Build an advect object for each quad
        for i_quad, quad in enumerate(self.patch_set.quads):
            boundary = []
            for edge in self.patch_set.edges[quad]:
                for coord in self.patch_set.vertices[edge][:-1]:
                    boundary.append(coord.tolist())
            boundary = torch.tensor(boundary, device=rc.device)
            transformation = BicubicPatch(boundary=boundary)

            # Initialize velocity and transformation based on first patch:
            if i_quad == 0:
                origin = transformation(torch.zeros((1, 2), device=rc.device))
                Rbasis = (transformation(torch.eye(2, device=rc.device)) - origin).T
                delta_Qfrac = torch.tensor(
                    [-1.0, -1.0] if diag else [1.0, 0.0], device=rc.device
                )
                delta_q = delta_Qfrac @ Rbasis.T

                # Initialize velocities (eventually should be in Material):
                init_angle = torch.atan2(delta_q[1], delta_q[0]).item()
                dtheta = 2 * np.pi / N_theta
                theta = torch.arange(N_theta, device=rc.device) * dtheta + init_angle
                v = v_F * torch.stack([theta.cos(), theta.sin()], dim=-1)

            new_patch = Advect(transformation=transformation, v=v, N=N)
            new_patch.origin = origin
            new_patch.Rbasis = Rbasis
            self.patches.append(new_patch)

    def apply_boundaries(self, patch_ind, patch, rho) -> torch.Tensor:
        """Apply all boundary conditions to `rho` and produce ghost-padded version."""
        non_ghost = patch.non_ghost
        ghost_l = patch.ghost_l
        ghost_r = patch.ghost_r
        out = torch.zeros(patch.rho_padded_shape, device=rc.device)
        out[non_ghost, non_ghost] = rho

        patch_adj = self.patch_set.adjacency[patch_ind].to(rc.cpu).numpy()
        in_slices = [
            (slice(None), ghost_l),
            (ghost_r, slice(None)),
            (slice(None), ghost_r),
            (ghost_l, slice(None)),
        ]
        out_slices = [
            (non_ghost, ghost_l),
            (ghost_r, non_ghost),
            (non_ghost, ghost_r),
            (ghost_l, non_ghost),
        ]
        BOUNDARY_FLIP_DIMS = {
            (0, 0): (1,),
            (0, 1): (0, 1),
            (0, 2): tuple[int](),
            (0, 3): (0,),
            (1, 0): (1,),
            (1, 1): (0, 1),
            (1, 2): tuple[int](),
            (1, 3): tuple[int](),
            (2, 0): tuple[int](),
            (2, 1): (0,),
            (2, 2): (0, 1),
            (2, 3): (0,),
            (3, 0): tuple[int](),
            (3, 1): tuple[int](),
            (3, 2): (0, 1),
            (3, 3): (0, 1),
        }

        # This logic is not correct yet (flips/transposes)
        # Check if each edge is reflecting, otherwise handle
        # ghost zone communication (16 separate cases)
        for i_edge, (other_patch_ind, other_edge) in enumerate(patch_adj):
            if other_patch_ind < 0:
                # TODO: handle reflecting boundaries
                # For now they will be sinks (hence pass)
                pass
            else:
                # Pass-through boundary:
                other_patch = self.patches[other_patch_ind]
                ghost_area = other_patch.rho_prev[in_slices[other_edge]]
                if (i_edge - other_edge) % 2:
                    ghost_area = ghost_area.swapaxes(0, 1)
                flip_dims = BOUNDARY_FLIP_DIMS[(i_edge, other_edge)]
                if flip_dims:
                    ghost_area = ghost_area.flip(dims=flip_dims)
                out[out_slices[i_edge]] = ghost_area
        return out

    # Geometry level time step
    def time_step(self):
        for patch in self.patches:
            patch.rho_prev = patch.rho.detach().clone()
        for i, patch in enumerate(self.patches):
            rho_half = patch.rho + patch.drho(
                0.5 * patch.dt, self.apply_boundaries(i, patch, patch.rho_prev)
            )
            patch.rho += patch.drho(patch.dt, self.apply_boundaries(i, patch, rho_half))


def equivalence_classes(pairs: torch.Tensor) -> torch.Tensor:
    """Given Npair x 2 array of index pairs that are equivalent,
    compute equivalence class numbers for each original index."""
    # Construct adjacency matrix:
    N = pairs.max() + 1
    i_pair, j_pair = pairs.T
    adjacency_matrix = torch.eye(N, device=rc.device)
    adjacency_matrix[i_pair, j_pair] = 1.0
    adjacency_matrix[j_pair, i_pair] = 1.0

    # Expand to indirect neighbors by repeated multiplication:
    n_non_zero_prev = torch.count_nonzero(adjacency_matrix)
    for i_mult in range(N):
        adjacency_matrix = adjacency_matrix @ adjacency_matrix
        n_non_zero = torch.count_nonzero(adjacency_matrix)
        if n_non_zero == n_non_zero_prev:
            break  # highest-degree connection reached
        n_non_zero_prev = n_non_zero

    # Find first non-zero entry of above (i.e. first equivalent index):
    is_first = torch.logical_and(
        adjacency_matrix.cumsum(dim=1) == adjacency_matrix, adjacency_matrix != 0.0
    )
    first_index = torch.nonzero(is_first)[:, 1]
    assert len(first_index) == N
    return torch.unique(first_index, return_inverse=True)[1]  # minimal class indices
