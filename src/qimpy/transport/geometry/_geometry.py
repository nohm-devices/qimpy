from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist

from qimpy import rc, log, TreeNode
from qimpy.io import CheckpointPath, InvalidInputException, CheckpointContext
from qimpy.mpi import (
    ProcessGrid,
    get_block_slices,
    TaskDivisionCustom,
    all_gather_padded,
    globalreduce,
)
from qimpy.profiler import StopWatch, stopwatch
from ..material import Material
from . import Mesh


class Geometry(TreeNode):
    """Cell-centered finite-volume geometry on a 1D/2D `Mesh`."""

    def __init__(
        self,
        *,
        material: Material,
        mesh: Mesh | dict | None = None,
        contacts: dict[str, dict],
        courant_number: float = 1.0,
        epsilon_sq: float = 0.0,
        faces_block_size: int = 0,
        compile: bool = False,
        save_rho: bool = False,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Parameters
        ----------
        mesh
            :yaml:`Specification of 2D or 1D mesh geometry for transport.`
        contacts
            :yaml:`Dictionary of contact names to parameters (match mesh markers).`
            Each value selects the contact kind: ``{dmu, vD}`` a fixed
            voltage/drift source, ``{floating: true}`` a zero-current probe, and
            ``{I_set: <current>}`` a current source (with optional ``vD``).
        courant_number
            :yaml:`Courant number for selecting the maximum stable time step.`
        epsilon_sq
            :yaml:`Flux-squared threshold on flux limiter.`
            Default of 0 yields a pure smooth limiter. Set to ~(mesh-scale variation)^2
            to not limit smooth variations and recover strict linearity preservation.
        faces_block_size
            :yaml:`Number of faces to process per block in the face flux limiter.`
            By default, process all the faces together, which is fastest on large-enough
            GPUs, but consumes more memory. Set to a smaller chunk size to reduce memory
            footprint, or to speed up on CPUs, where that may help with memory locality.
        compile
            :yaml:`Whether to compile the face flux limiter kernel.`
            Speeds up sequential elemntise operations by kernel fusion when possible.
        """
        TreeNode.__init__(self)
        self.group = process_grid.get_group("r")
        self.material = material
        self.add_child("mesh", Mesh, mesh, checkpoint_in, group=self.group)
        self.save_rho = save_rho
        self.epsilon_sq = float(epsilon_sq)
        self.faces_block_size = faces_block_size

        self._mpi = self.group.size() > 1
        v = material.transport_velocity  # (Nk, 2)
        geom = self.mesh.get_fv_geom(dtype=v.dtype)
        cell_division = self.mesh.cell_division
        self.geom = geom
        self.Nk = v.shape[0]
        self.K = int(geom.area.shape[0])  # local cells including padding
        self._nf = int(geom.recon.shape[1])  # faces/cell: 3 (tri) or 2 (1D)

        # Precompute velocity-weighted, area-scaled edge operators (constant):
        #   into eL: -a*elen/area_L,  into eR: +a*elen/area_R,  a = v_c.n
        a_int = geom.en @ v.t()  # (Ne, Nk)
        a_bnd = geom.bn @ v.t()  # (Nb, Nk)
        inv_area = 1 / geom.area
        maskL = a_int > 0
        self.is_outward = a_bnd > 0
        wL = -a_int * (geom.elen * inv_area[geom.eL])[:, None]
        wR = a_int * (geom.elen * inv_area[geom.eR])[:, None]
        wB = -a_bnd * (geom.blen * inv_area[geom.bcell])[:, None]
        ik = torch.arange(self.Nk, device=rc.device)
        idx_in = (
            ik
            + self.Nk
            * torch.cat(
                (
                    torch.where(maskL, geom.eLF[:, None], geom.eRF[:, None]).tile(2, 1),
                    geom.bF[:, None].expand(-1, self.Nk),
                )
            )
        ).flatten()
        idx_out = (
            ik + self.Nk * torch.cat((geom.eL, geom.eR, geom.bcell))[:, None]
        ).flatten()
        values = torch.cat((wL.flatten(), wR.flatten(), wB.flatten()))
        sel_mine = torch.where(idx_out < cell_division.n_mine * self.Nk)[0]
        n_recon = cell_division.n_recon
        self.v_dot_del = torch.sparse_coo_tensor(
            indices=torch.stack((idx_out[sel_mine], idx_in[sel_mine])),
            values=values[sel_mine],
            size=(len(geom.area) * self.Nk, n_recon * self._nf * self.Nk),
        ).to_sparse_csr()
        self.dt_max = courant_number / globalreduce.max(values, dist.group.WORLD).item()
        log.info(f"Set dt_max = {self.dt_max:.4g} for Courant number {courant_number}")

        self.collision_enabled = True
        self.boundary = Boundary(self, contacts)

        # Initialize flux collection on faces:
        self.flux_indices = material.get_flux_indices()
        if self.flux_indices:
            to_global = torch.from_numpy(cell_division.global_index).to(rc.device)
            eL = to_global[geom.eL]
            eR = to_global[geom.eR]
            bcell = to_global[geom.bcell]
            edge_cells = torch.cat(
                [
                    torch.stack([eL, eR], 1)[: geom.n_interior_mine],
                    torch.stack([bcell, -torch.ones_like(bcell)], 1),
                ]
            )
            edge_normal = torch.cat((geom.en[: geom.n_interior_mine], geom.bn))
            edge_len = torch.cat((geom.elen[: geom.n_interior_mine], geom.blen))
            n_edges_mine = len(edge_cells)
            edge_div = TaskDivisionCustom(n_mine=n_edges_mine, group=self.group)

            # Store collected global-index versions of above for checkpoint writing:
            sizes = edge_div.n_each_custom
            self._edge_cells = all_gather_padded(edge_cells, sizes, self.group).cpu()
            self._edge_normal = all_gather_padded(edge_normal, sizes, self.group).cpu()
            self._edge_len = all_gather_padded(edge_len, sizes, self.group).cpu()
            self.edge_division = edge_div

            # Construct flux matrix (selects appropriate direction and multiplies v):
            mine = slice(0, geom.n_interior_mine)
            idx_in = (
                ik
                + self.Nk
                * torch.cat(
                    (
                        torch.where(maskL, geom.eLF[:, None], geom.eRF[:, None])[mine],
                        geom.bF[:, None].expand(-1, self.Nk),
                    )
                )
            ).flatten()
            idx_out = (
                ik + self.Nk * torch.arange(n_edges_mine, device=rc.device)[:, None]
            ).flatten()
            values = (
                self.material.wk
                * torch.cat(
                    (
                        (a_int * geom.elen[:, None])[mine],
                        a_bnd * geom.blen[:, None],
                    )
                )
            ).flatten()
            self.v_multiply_select = torch.sparse_coo_tensor(
                indices=torch.stack((idx_out, idx_in)),
                values=values,
                size=(n_edges_mine * self.Nk, n_recon * self._nf * self.Nk),
            ).to_sparse_csr()

            # Set up boundary probes:
            i_boundary_all = self.mesh.boundary_faces[:, -1]
            assert i_boundary_all.min() >= -1
            n_boundaries = int(i_boundary_all.max()) + 1  # marked boundary regions
            probe_edges = torch.where(geom.bmark >= 0)[0]
            idx_in = torch.arange(len(probe_edges) * self.Nk, device=rc.device)
            idx_out = (ik + self.Nk * geom.bmark[probe_edges, None]).flatten()
            values = (
                self.material.wk * a_bnd[probe_edges] * geom.blen[probe_edges, None]
            ).flatten()
            self.boundary_flux_op = torch.sparse_coo_tensor(
                indices=torch.stack((idx_out, idx_in)),
                values=values,
                size=(n_boundaries * self.Nk, len(probe_edges) * self.Nk),
            ).to_sparse_csr()
            # Scalars at surface cells:
            idx_in = torch.arange(len(probe_edges), device=rc.device)
            idx_out = geom.bmark[probe_edges]
            w_faces = geom.blen[probe_edges]
            w_tot = torch.zeros(n_boundaries, dtype=w_faces.dtype, device=rc.device)
            w_tot.index_add_(0, idx_out, w_faces)
            dist.all_reduce(w_tot, group=self.group)
            self.boundary_scalar_op = torch.sparse_coo_tensor(
                indices=torch.stack((idx_out, idx_in)),
                values=w_faces / w_tot[idx_out],
                size=(n_boundaries, len(probe_edges)),
            ).to_sparse_csr()
            self.boundary_probe = Boundary(self, contacts, probe_edges)

        # Set up cell region (interior) probes:
        n_mine = cell_division.n_mine
        w_cells = geom.area[:n_mine]
        i_region_all = self.mesh.cells[:, -1]
        assert i_region_all.min() >= -1
        n_regions = int(i_region_all.max()) + 1
        i_region = torch.from_numpy(
            i_region_all[cell_division.global_index[:n_mine]]
        ).to(rc.device)
        probe_cells = torch.where(i_region >= 0)[0]
        i_region = i_region[probe_cells]
        w_cells = w_cells[probe_cells]
        w_tot = torch.zeros(n_regions, dtype=w_cells.dtype, device=rc.device)
        w_tot.index_add_(0, i_region, w_cells)
        dist.all_reduce(w_tot, group=self.group)
        n_probe = len(probe_cells)
        self.probe_cells = probe_cells
        self.probe_mat_cell = torch.sparse_coo_tensor(
            indices=torch.stack((i_region, torch.arange(n_probe, device=rc.device))),
            values=w_cells / w_tot[i_region],
            size=(n_regions, n_probe),
        ).to_sparse_csr()  # multiply by per-cell observables to get region averages

        # Compile the face limiter which involves several elemntwise operations:
        if compile:
            mode = os.environ.get("QIMPY_COMPILE_MODE", "max-autotune")
            if mode == "jit-script":
                self._face_limiter = torch.jit.script(_face_limiter)
            else:
                self._face_limiter = torch.compile(_face_limiter, mode=mode)
        else:
            self._face_limiter = _face_limiter  # eager mode

        rho0 = getattr(material, "rho0", None)
        if checkpoint_in and checkpoint_in.member("rho"):
            cp, path = checkpoint_in  # warm start (serial)
            rho = torch.as_tensor(
                np.array(cp[f"{path}/rho"]), device=rc.device, dtype=v.dtype
            )
            if tuple(rho.shape) != (self.K, self.Nk):
                raise InvalidInputException(
                    f"checkpoint rho shape {tuple(rho.shape)} != ({self.K}, {self.Nk})"
                )
        elif rho0 is not None:
            rho = rho0.flatten().to(rc.device, v.dtype)[None, :].repeat(self.K, 1)
        else:
            rho = torch.zeros(self.K, self.Nk, device=rc.device, dtype=v.dtype)
        self.rho = rho
        self.stash = Stash()
        self.stash_probe = Stash()  # coarse-grained, more frequent version
        self.material.initialize_fields(self.rho, {}, id(self))

    def to_faces(
        self, rho: torch.Tensor, t: float, boundary_only: bool = False
    ) -> torch.Tensor:
        """Reconstruct face values from cell-centred values and boundary conditions.
        Result is flattened over the cell and face dimensions.
        If `boundary_only`, result is computed only for marked boundary faces."""
        # Update ghost regions if needed:
        cell_division = self.mesh.cell_division
        cell_division.exchange(rho)

        # Reconstruct rho at faces from rho at cell centroids:
        g = self.geom
        if boundary_only:
            sel = self.boundary_probe.boundary_cells
            rho_faces = self._face_limiter(
                rho[sel], rho, g.nbr[sel], g.recon[sel], self.epsilon_sq
            )
            # Apply boundary conditions and return only boundary faces:
            return self.boundary_probe.apply(rho_faces.flatten(0, 1), t, update=False)
        else:
            n_recon = cell_division.n_recon
            rho_faces = rho.new_zeros(n_recon, self._nf, self.Nk)
            block_size = self.faces_block_size if self.faces_block_size else n_recon
            for sel in get_block_slices(n_recon, block_size):
                rho_faces[sel] = self._face_limiter(
                    rho[sel], rho, g.nbr[sel], g.recon[sel], self.epsilon_sq
                )
            # Apply boundary conditions and return all faces:
            rho_faces = rho_faces.flatten(0, 1)
            self.boundary.apply(rho_faces, t, update=True)
            return rho_faces

    def get_flux(
        self, t: float, boundary_only: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute flux on all faces, or only boundary regions if `boundary_only`.
        Also return `rho_faces`, if needed for calculating observables on faces."""
        rho_faces = self.to_faces(self.rho, t, boundary_only)
        v_op = self.boundary_flux_op if boundary_only else self.v_multiply_select
        v_rho = v_op @ rho_faces.flatten()
        flux_op = self.material.get_observables(t)[self.flux_indices].T
        flux = (v_rho.view(-1, self.Nk) @ flux_op).contiguous()
        dist.all_reduce(flux, group=(None if boundary_only else self.material.group))
        return flux, rho_faces

    def rho_dot(self, rho: torch.Tensor, t: float) -> torch.Tensor:
        self.material.constrain(rho)
        watch = StopWatch("Geometry.rho_dot")
        out = (self.v_dot_del @ self.to_faces(rho, t).flatten()).view(rho.shape)
        watch.stop()
        n_mine = self.mesh.cell_division.n_mine
        if self.collision_enabled:
            out[:n_mine] += self.material.rho_dot(rho[:n_mine], t, id(self))
        return out

    def collision_dot(self, rho: torch.Tensor, t: float) -> torch.Tensor:
        """The material collision term ALONE, with no spatial RHS.
        Used by operator splitting (``collision_interval > 1``)."""
        out = torch.zeros_like(rho)
        n_mine = self.mesh.cell_division.n_mine
        out[:n_mine] += self.material.rho_dot(rho[:n_mine], t, id(self))
        return out

    @property
    def rho(self) -> torch.Tensor:
        return self._rho

    @rho.setter
    def rho(self, rho_new: torch.Tensor) -> None:
        self._rho = rho_new
        self.material.constrain(self._rho)

    @stopwatch
    def update_probes(self, i_step: int, t: float) -> None:
        """Stash coarse-grained probes on defined cell regions and contacts."""
        self.stash_probe.i_step.append(i_step)
        self.stash_probe.t.append(t)
        # Interior cell regions:
        obserables = self.material.measure_observables(self.rho[self.probe_cells], t)
        region_averages = (self.probe_mat_cell @ obserables).contiguous()
        dist.all_reduce(region_averages, group=self.group)
        if self.flux_indices:
            flux, rho_faces = self.get_flux(t, boundary_only=True)
            observables = self.material.measure_observables(rho_faces, t)
            surface_averages = (self.boundary_scalar_op @ observables).contiguous()
            dist.all_reduce(surface_averages, group=self.group)
            region_averages = torch.cat((region_averages, surface_averages))
            self.stash_probe.flux.append(flux.detach().cpu())
        self.stash_probe.observables.append(region_averages.detach().cpu())

    @stopwatch
    def update_observables(self, i_step: int, t: float) -> None:
        """Stash observables (and related) for output in next checkpoint."""
        rho_mine = self.rho[: self.mesh.cell_division.n_mine]
        obserables = self.material.measure_observables(rho_mine, t)
        self.stash.i_step.append(i_step)
        self.stash.t.append(t)
        self.stash.observables.append(obserables.detach().cpu())
        if self.flux_indices:
            self.stash.flux.append(self.get_flux(t)[0].detach().cpu())
        self.material.update_stash(rho_mine, self.mesh.cell_division)

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        checkpoint, path = cp_path
        assert checkpoint is not None
        names = self.material.get_observable_names()
        saved = [
            cp_path.write("flux_indices", np.array(self.flux_indices)),
            cp_path.write_str("observable_names", ",".join(names)),
            cp_path.write("t", np.array(self.stash.t)),
            cp_path.write("i_step", np.array(self.stash.i_step)),
            "observables",
            cp_path.write("t_probe", np.array(self.stash_probe.t)),
            cp_path.write("i_step_probe", np.array(self.stash_probe.i_step)),
            "probes",
        ]

        # Observables:
        n_stash = len(self.stash.t)
        div = self.mesh.cell_division
        dset = cp_path.create_dataset(
            "observables", (n_stash, div.n_tot, len(names)), np.float64
        )
        checkpoint.write_slice(
            dset, (0, div.i_start, 0), torch.stack(self.stash.observables)
        )
        if self.flux_indices:
            # Flux observables live on edges: geometry + face-normal flux per frame.
            saved += [
                cp_path.write("edge_cells", self._edge_cells),
                cp_path.write("edge_normal", self._edge_normal),
                cp_path.write("edge_len", self._edge_len),
                "edge_flux",
            ]
            edge_div = self.edge_division
            flux = torch.stack(self.stash.flux)
            dset = cp_path.create_dataset(
                "edge_flux", (n_stash, edge_div.n_tot, len(flux)), np.float64
            )
            checkpoint.write_slice(dset, (0, edge_div.i_start, 0), flux)
        self.stash = Stash()  # reset now-written cached observables

        # Probes:
        cp_path.write("probes", torch.stack(self.stash_probe.observables))
        cp_path.write("probe_flux", torch.stack(self.stash_probe.flux))
        self.stash_probe = Stash()  # reset now-written cached probes

        if self.save_rho:
            # Optional full state output (large) for continuing calculations:
            rho_mine = self.rho[: div.n_mine].detach()
            dset = cp_path.create_dataset("rho", (div.n_tot, self.Nk), rho_mine.dtype)
            checkpoint.write_slice(dset, (div.i_start, 0), rho_mine)
            saved.append("rho")

        return saved


def _face_limiter(
    rho_sel: torch.Tensor,
    rho: torch.Tensor,
    neighbors: torch.Tensor,
    recon: torch.Tensor,
    epsilon_sq: float,
) -> torch.Tensor:
    """Venkatakrishnan-limited face values for a subset of cells with values `rho_sel`,
    where `rho` is the full data, `neighbors` lists the neighboring cell indices for
    the selected cells, and `reconstruct` is the fused gradient->face operator.
    Limiter function applied per face is:
        phi = (D1^2 + 2 D1 d + epsilon^2) / (D1^2 + D1 d + 2 d^2 + epsilon^2),
    where d = rho_face - rho, and D1 is the same-sign headroom to rho differences."""
    drho = rho[neighbors] - rho_sel[:, None]
    d = recon @ drho
    # Compute limiter (in [0, 1]) for the gradients on each face:
    lo, hi = drho.aminmax(dim=1, keepdim=True)
    lo.clip_(max=0.0)
    hi.clip_(min=0.0)
    D1 = torch.where(d >= 0, hi, lo)
    common = D1.square() + epsilon_sq
    dD1 = d * D1
    phi = (common + 2 * dD1) / (common + dD1 + 2 * d.square())
    phi.nan_to_num_(1.0)  # Handle possible NANs from d underflow
    phi.clamp_(max=1.0)
    limiter = phi.amin(dim=1, keepdim=True)
    return rho_sel[:, None] + limiter * d


@dataclass
class Stash:
    """Stash for probes and observables that are collated into checkpoints."""

    i_step: list[int] = field(default_factory=list)
    t: list[float] = field(default_factory=list)
    observables: list[torch.Tensor] = field(default_factory=list)
    flux: list[torch.Tensor] = field(default_factory=list)


class Boundary:
    """Boundary operators and indices for the entire boundary, or a selection of it."""

    contacts: list[tuple[torch.Tensor, None | Callable[[float], torch.Tensor]]]
    wall: tuple[torch.Tensor, None | Callable[[torch.Tensor], torch.Tensor]]
    is_outward: torch.Tensor  #: whether each face, k velocity is outwards
    boundary_cells: torch.Tensor | None  #: subset of cells at which rho_faces is needed
    boundary_faces: torch.Tensor  #: flattened index of boundary edges into rho_faces

    def __init__(
        self,
        geometry: Geometry,
        contacts: dict[str, dict],
        selection: torch.Tensor | None = None,
    ) -> None:
        g = geometry.geom
        names = geometry.mesh.boundary_names
        material = geometry.material
        bmark = g.bmark
        bn = g.bn
        self.boundary_cells = None
        self.boundary_faces = g.bF
        if selection is not None:
            bmark = bmark[selection]
            bn = bn[selection]
            # Find the cells for which rho_faces is needed for `selection`:
            cells = g.bcell[selection]
            self.boundary_cells, i_cell = torch.unique(cells, return_inverse=True)
            n_faces = geometry._nf
            i_face = g.bF[selection] - n_faces * cells
            self.boundary_faces = i_cell * n_faces + i_face

        edge_used = torch.full((len(bmark),), False, device=rc.device, dtype=torch.bool)
        self.contacts = []
        for name, params in contacts.items():
            indices = torch.where(bmark == names.index(name))[0]
            if len(indices):
                edge_used[indices] = True
                contactor = material.get_contactor(bn[indices], **params)
                self.contacts.append((indices, contactor))

        # Mark all boundary edges not in a contact as reflecting:
        indices = torch.where(~edge_used)[0]
        reflector = material.get_reflector(bn[indices]) if len(indices) else None
        self.wall = (indices, reflector)
        self.is_outward = (bn @ material.transport_velocity.T) > 0

    def apply(self, rho_faces: torch.Tensor, t: float, update: bool) -> torch.Tensor:
        """Apply boundary conditions to `rho_faces`, modifying it in place
        if `update`. Otherwise, return the updated component alone."""
        bf = self.boundary_faces
        rho_outward = rho_faces[bf]  # boundary faces, valid outwards
        # Apply reflections and contacts:
        rho_inward = rho_outward.detach().clone()  # prepare inward valid version
        indices, reflector = self.wall
        if reflector is not None:
            rho_inward[indices] = reflector(rho_outward[indices])
        for indices, contactor in self.contacts:
            rho_inward[indices] = contactor(t)
        # Set back to faces / return:
        rho_faces_bf = torch.where(self.is_outward, rho_outward, rho_inward)
        if update:
            rho_faces[bf] = rho_faces_bf
        return rho_faces_bf
