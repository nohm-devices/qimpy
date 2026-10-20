from __future__ import annotations
from dataclasses import dataclass

from scipy.sparse import csr_array
import numpy as np
import torch
import torch.distributed as dist

from qimpy import rc, TreeNode
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.mpi import TaskDivisionCustom
from . import mesh_source


class Mesh(TreeNode):
    """Definition of mesh geometry (dimension, d = 2 or 1)"""

    source: mesh_source.Options  #: Method by which mesh is constructed
    vertices: np.ndarray  #: vertex coordinates (Nv, 2)
    cells: np.ndarray  #: vertex indices and region ID in cells (Nc, d+2)
    region_names: list[str]  #: names of interior regions (for cell IDs >= 0)
    boundary_faces: np.ndarray  #: sorted vertex indices, ID of boundary faces (Nb, d+1)
    boundary_names: list[str]  #: names of boundary regions (for face IDs >= 0)
    cell_permutation: np.ndarray | None  #: Reordering of cells relative to the source

    # Calculated members:
    cell_division: CellDivision  #: split of cells over processes
    faces: np.ndarray  #: Vertex indices by cell and face (Nc, d+1, d)
    interior: np.ndarray  #: Pair of cell and face indices by interior face (Ni, 2, 2)
    boundary: np.ndarray  #: Cell, face and region indices by boundary face (Nb, 3)

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        file: mesh_source.File | dict | None = None,
        line: mesh_source.Line | dict | None = None,
        rect: mesh_source.Rect | dict | None = None,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """
        Initialize mesh geometry.

        Parameters
        ----------
        file
            :yaml:`Load mesh from HDF5 file.`
            Only one source for the mesh must be specified.
        line
            :yaml:`Construct uniform mesh for a 1D domain.`
            Only one source for the mesh must be specified.
        rect
            :yaml:`Construct triangular mesh for a 2D rectangular domain.`
            Only one source for the mesh must be specified.
        """
        super().__init__()
        self.cell_permutation = None
        self.add_child_one_of(
            "source",
            checkpoint_in,
            TreeNode.ChildOptions("file", mesh_source.File, file, mesh=self),
            TreeNode.ChildOptions("line", mesh_source.Line, line, mesh=self),
            TreeNode.ChildOptions("rect", mesh_source.Rect, rect, mesh=self),
            have_default=False,
        )
        if checkpoint_in:
            self.load(checkpoint_in)

        self._initialize_face_maps()
        self.cell_division = self._partition_cells(group)

    def load(self, cp: CheckpointPath) -> None:
        self.vertices = cp.read_np("vertices")
        self.cells = cp.read_np("cells")
        self.region_names = cp.read_str_list("region_names")
        self.boundary_faces = cp.read_np("boundary_faces")
        self.boundary_names = cp.read_str_list("boundary_names")
        self.cell_permutation = cp.read_optional_np("cell_permutation")

    def save(self, cp: CheckpointPath) -> list[str]:
        """Save within h5 file and return names of saved variables."""
        saved_list = [
            cp.write("vertices", self.vertices),
            cp.write("cells", self.cells),
            cp.write_str("region_names", ",".join(self.region_names)),
            cp.write("boundary_faces", self.boundary_faces),
            cp.write_str("boundary_names", ",".join(self.boundary_names)),
        ]
        if self.cell_permutation is not None:
            saved_list.append(cp.write("cell_permutation", self.cell_permutation))
        return saved_list

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        return self.save(cp_path)

    @property
    def is1D(self) -> bool:
        return self.cells.shape[1] == 3  # cells coontain d + 1 vertices, 1 region index

    def _partition_cells(self, group: dist.ProcessGroup) -> CellDivision:
        """Optimally partition the mesh cells for communication over `group`.
        This reorders `cells` such that contiguous ranges of cells can be owned by
        each process within `group`, and returns a `CellDivision` instance to
        track the division and communicate ghost cells across processes.
        Correspondingly, `cell_permutation` tracks any reordering of the cells."""
        size = group.size()
        if size == 1:
            counts = np.array([len(self.cells)])  # no partitioning required
            return CellDivision(self, counts, group)

        # Compute partition on overall head process:
        # (Partition must be stable across process grid, not just the "r" dimension.)
        partition = np.empty(len(self.cells), dtype=np.int32)
        if rc.is_head:
            try:
                import pymetis

                adj = self.get_cell_adjacency_vertex()
                csr_adj = pymetis.CSRAdjacency(adj.indptr, adj.indices)
                partition = np.asarray(pymetis.part_graph(size, csr_adj)[1], np.int32)
            except ImportError:
                # Fallback: divide equally along longest axis:
                centroids = self.vertices[self.cells[:, :-1]].mean(axis=1)
                long_axis = np.ptp(centroids, axis=0).argmin()
                order = np.argsort(centroids[:, long_axis], kind="stable")
                partition[order] = np.minimum(
                    (np.arange(len(order)) * size) // len(order), size - 1
                )

        # Synchronize partition across all processes:
        partition_t = torch.from_numpy(partition).to(rc.device)
        dist.broadcast(partition_t, src=0)  # on world, not just on group
        partition = partition_t.to(rc.cpu).numpy()

        # Determine permutation and bounds:
        permutation = np.argsort(partition, kind="stable")  # group cells by rank
        counts = np.bincount(partition, minlength=size)

        # Apply and track permutation:
        self.cells = self.cells[permutation]
        self.faces = self.faces[permutation]
        inv_permutation = np.zeros_like(permutation)
        inv_permutation[permutation] = np.arange(len(permutation))
        self.interior[..., 0] = inv_permutation[self.interior[..., 0]]
        self.boundary[:, 0] = inv_permutation[self.boundary[:, 0]]
        if self.cell_permutation is None:
            self.cell_permutation = permutation
        else:
            self.cell_permutation = self.cell_permutation[permutation]
        return CellDivision(self, counts, group)

    def get_cell_adjacency_vertex(self) -> csr_array:
        """Adjacency matrix of cells that share a vertex."""
        cells = self.cells[:, :-1]
        cell_to_vertex = csr_array(
            (
                np.ones_like(cells).flatten(),
                (np.repeat(np.arange(len(cells)), cells.shape[1]), cells.flatten()),
            ),
        )
        return make_adjacency(cell_to_vertex)

    def get_cell_adjacency_face(self) -> csr_array:
        """Adjacency matrix of cells that share an face."""
        cell_pairs = self.interior[..., 0]
        i_cell, j_cell = np.vstack((cell_pairs, cell_pairs[:, ::-1])).T  # both orders
        n_cells = len(self.cells)
        return csr_array((np.ones_like(i_cell), (i_cell, j_cell)), (n_cells, n_cells))

    def _initialize_face_maps(self) -> None:
        """Initialize interior and boundary faces, mapped to cells."""
        n_faces_per_cell = self.cells.shape[1] - 1  # for each vertex dropped from cell
        faces = np.stack(
            [
                np.roll(self.cells[:, :-1], -i_drop, axis=1)[:, 1:]
                for i_drop in range(n_faces_per_cell)
            ],
            axis=1,
        )
        stride = len(self.vertices) ** np.arange(faces.shape[-1])[::-1]
        face_ids = np.sort(faces, axis=-1) @ stride  # equal for all vertex orders
        boundary_face_ids = np.sort(self.boundary_faces[:, :-1], axis=-1) @ stride
        self.faces = faces

        # Find unique face IDs and counts:
        unique_ids, unique_indices, unique_inverse, counts = np.unique(
            face_ids, return_index=True, return_inverse=True, return_counts=True
        )
        assert counts.max() <= 2
        unique_rank = np.ones_like(unique_inverse)  # rank within equivalence class
        unique_rank[unique_indices] = 0  # and others are 1 because count <= 2
        face_map = np.zeros_like(unique_inverse, shape=(len(unique_ids), 2, 2))
        face_map[unique_inverse, unique_rank] = np.stack(
            np.meshgrid(
                np.arange(faces.shape[0]), np.arange(faces.shape[1]), indexing="ij"
            ),
            axis=-1,
        ).reshape(
            -1, 2
        )  # cell and face within cell, for each copy of each unique face

        # Split face map into interior and boundary
        self.interior = face_map[np.where(counts == 2)[0]]
        sel_boundary = np.where(counts == 1)[0]

        # Find boundary region index for each boundary face:
        i_unique_boundary = np.searchsorted(unique_ids, boundary_face_ids)
        boundary_region_ids = np.full(len(unique_ids), -1)
        boundary_region_ids[i_unique_boundary] = self.boundary_faces[:, -1]
        self.boundary = np.hstack(
            (face_map[sel_boundary, 0], boundary_region_ids[sel_boundary, None])
        )

    def get_fv_geom(self, *, dtype: torch.dtype = torch.float64) -> FVGeom:
        """Initialize finite-volume parameters for mesh."""
        # Compute cell area and centroid:
        cell_division = self.cell_division
        cells = self.cells[cell_division.global_index, :-1]  # local, ignore region ID
        cell_vertices = self.vertices[cells]
        if self.is1D:
            area = np.linalg.norm(cell_vertices[:, 1] - cell_vertices[:, 0], axis=-1)
        else:
            area = 0.5 * np.linalg.det(cell_vertices[:, 1:] - cell_vertices[:, :1])
            if np.any(area <= 0.0):
                raise ValueError("Areas must be positive (counter-clockwise triangles)")
        centroid = cell_vertices.mean(axis=1)

        # Identify interior and boundary faces, and map to local cell indices:
        faces = self.faces[cell_division.global_index]  # local cells (with padding)
        interior = np.copy(self.interior)
        boundary = np.copy(self.boundary)
        interior[..., 0] = cell_division.local_index[interior[..., 0]]
        boundary[:, 0] = cell_division.local_index[boundary[:, 0]]

        # Filter interior faces so that both cells local and at least one owned:
        interior = interior[(interior[..., 0] >= 0).all(axis=1)]  # both cells local
        interior = interior[(interior[..., 0] < cell_division.n_mine).any(axis=1)]
        # Define face ownership based on first cell owned, and sort by ownership:
        interior = interior[np.lexsort(interior[:, ::-1, 0].T)]
        n_interior_mine = int(np.searchsorted(interior[:, 0, 0], cell_division.n_mine))
        (kL, fL), (kR, fR) = np.transpose(interior, (1, 2, 0))

        # Filter boundary faces so that corresponding cell is owned:
        sel = (boundary[:, 0] >= 0) & (boundary[:, 0] < cell_division.n_mine)
        boundary = boundary[sel]
        bk, bf, bmark = boundary.T

        # Compute face properties:
        face_vertices = self.vertices[faces]
        face_offset = face_vertices.mean(axis=2) - centroid[:, None]  # from centroid
        if self.is1D:
            face_weights = np.ones_like(face_offset[..., 0])
            face_normals = face_offset / np.linalg.norm(
                face_offset, axis=-1, keepdims=True
            )
        else:
            face_vectors = face_vertices[:, :, 1] - face_vertices[:, :, 0]
            face_weights = np.linalg.norm(face_vectors, axis=2)
            face_normals = (
                np.stack([face_vectors[..., 1], -face_vectors[..., 0]], axis=2)
                / face_weights[..., None]
            )

        # Inverse-distance-weighted least-squares gradient operator per cell:
        adjacency = self.get_cell_adjacency_vertex()
        neighbor_indices = cell_division.local_index[adjacency.indices]  # local index
        index_ranges = adjacency.indptr  # but index ranges by global index
        Nmax = np.diff(index_ranges).max()
        n_cells_recon = cell_division.n_recon
        nbr = np.arange(n_cells_recon)[:, None].repeat(Nmax, axis=1)  # pad with self
        grad_op = np.zeros((n_cells_recon, 2, Nmax))
        Nmin = 1 if self.is1D else 2
        for i, i_global in enumerate(cell_division.global_index[:n_cells_recon]):
            js = neighbor_indices[index_ranges[i_global] : index_ranges[i_global + 1]]
            if len(js) < Nmin:
                continue
            nbr[i, : len(js)] = js
            D = centroid[js] - centroid[i]
            w = 1.0 / np.linalg.norm(D, axis=1)
            grad_op[i, :, : len(js)] = np.linalg.pinv(w[:, None] * D) * w[None, :]

        # Fuse gradient + centroid -> face offsets to a single matrix multiply:
        recon = np.einsum("kfx, kxg -> kfg", face_offset[:n_cells_recon], grad_op)

        def t(a, long=False):
            return torch.tensor(
                np.ascontiguousarray(a),
                device=rc.device,
                dtype=torch.long if long else dtype,
            )

        n_cell_faces = 2 if self.is1D else 3
        return FVGeom(
            area=t(area),
            eL=t(kL, long=True),
            eR=t(kR, long=True),
            eLF=t(kL * n_cell_faces + fL, long=True),
            eRF=t(kR * n_cell_faces + fR, long=True),
            en=t(face_normals[kL, fL]),
            elen=t(face_weights[kL, fL]),
            bcell=t(bk, long=True),
            bF=t(bk * n_cell_faces + bf, long=True),
            bmark=t(bmark, long=True),
            bn=t(face_normals[bk, bf]),
            blen=t(face_weights[bk, bf]),
            nbr=t(nbr, long=True),
            recon=t(recon),
            n_interior_mine=n_interior_mine,
        )


@dataclass
class FVGeom:
    """Finite-volume geometry parameters for mesh."""

    area: torch.Tensor  #: Cell areas (interval lengths in 1D)
    eL: torch.Tensor  #: First cell index for interior faces
    eR: torch.Tensor  #: Second cell index for interior faces
    eLF: torch.Tensor  #: Flattened first cell-face index for interior faces
    eRF: torch.Tensor  #: Flattened second cell-face index for interior faces
    en: torch.Tensor  #: Outward normal directions for each face (in first cell)
    elen: torch.Tensor  #: Face areas
    bcell: torch.Tensor  #: Cell index for boundary faces
    bF: torch.Tensor  #: Flattened cell-face index for boundary faces
    bmark: torch.Tensor  #: Boundary region IDs for boundary faces
    bn: torch.Tensor  #: Outward normal directions for boundary faces
    blen: torch.Tensor  #: Boundary face areas
    nbr: torch.Tensor  #: Neighbor-indices by cell (n_cells, Nmax)
    recon: torch.Tensor  #: Face-increment  operator (n_cells, 3, Nmax)
    n_interior_mine: int  #: Number of interior faces owned by current cell


def make_adjacency(x_to_y: csr_array) -> csr_array:
    """Construct adjacency matrix of x based on shared property y,
    given the mapping from x to y."""
    n_x = x_to_y.shape[0]
    adjacency_coo = (x_to_y @ x_to_y.T).tocoo()
    row = adjacency_coo.row
    col = adjacency_coo.col
    sel = np.where(row != col)[0]  # ignore trivial diagonals
    adjacency = csr_array((np.ones_like(sel), (row[sel], col[sel])), (n_x, n_x))
    adjacency.sort_indices()
    return adjacency


class CellDivision(TaskDivisionCustom):
    """Divide cells and communicate ghosts between processes."""

    def __init__(
        self, mesh: Mesh, counts: np.ndarray, group: dist.ProcessGroup
    ) -> None:
        super().__init__(n_mine=counts[group.rank()], group=group)
        self.group = group
        adj_vert = mesh.get_cell_adjacency_vertex()
        adj_face = mesh.get_cell_adjacency_face()

        def ring(cells: np.ndarray, adj: csr_array) -> np.ndarray:
            """1-ring vertex closure of a set of cells (cells + their neighbours)."""
            v = np.zeros(self.n_tot)
            v[cells] = 1
            v += adj @ v
            return np.where(v)[0]

        owned = np.arange(self.i_start, self.i_stop)
        ring1 = ring(owned, adj_face)  # cells to reconstruct (+face-neighbors)
        ring2 = ring(ring1, adj_vert)  # cells accessed overall (+vertex-neighbors)
        ghosts = ring2[~self.belongs(self.i_proc, ring2)]  # global indices from others
        self.n_recon = len(ring1)  # number of local cells to reconstruct

        # Collect global cell indices of owned, ring1 and ring2 on current process
        ring1_only = ring1[~np.isin(ring1, owned)]
        ring2_only = ring2[~np.isin(ring2, ring1)]
        self.global_index = np.concatenate((owned, ring1_only, ring2_only))

        # Inverse map from all global cell indices to local ones (-1 for untouched ones)
        self.local_index = np.full(self.n_tot, -1)
        self.local_index[self.global_index] = np.arange(len(ring2))

        # Plan communications:
        recv: list[np.ndarray] = []
        send: list[np.ndarray] = []
        for j_proc in range(self.n_procs):
            if j_proc == self.i_proc:
                recv.append(np.empty(0, dtype=int))
                send.append(np.empty(0, dtype=int))
            else:
                # Receive each ghost from its owning rank:
                recv.append(self.local_index[ghosts[self.belongs(j_proc, ghosts)]])
                # Send the owned cells that other ranks need:
                owned_q = np.arange(self.n_prev[j_proc], self.n_prev[j_proc + 1])
                ring2_q = ring(ring(owned_q, adj_face), adj_vert)
                ghost_q = ring2_q[~self.belongs(j_proc, ring2_q)]
                send.append(self.local_index[owned[np.isin(owned, ghost_q)]])
        self.recv_index = torch.from_numpy(np.concatenate(recv)).to(rc.device)
        self.send_index = torch.from_numpy(np.concatenate(send)).to(rc.device)
        self.recv_counts = [len(idx) for idx in recv]
        self.send_counts = [len(idx) for idx in send]

    def exchange(self, u: torch.Tensor) -> None:
        """Fill this rank's ghost rows of ``u`` (K, Nk) with their owners' values."""
        if self.n_procs == 1:
            return
        send_buf = u[self.send_index].contiguous()
        recv_buf = torch.empty(
            (len(self.recv_index), u.shape[1]), dtype=u.dtype, device=u.device
        )
        dist.all_to_all_single(
            recv_buf, send_buf, self.recv_counts, self.send_counts, group=self.group
        )
        u[self.recv_index] = recv_buf
