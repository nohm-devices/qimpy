"""Cell-centered 2nd-order finite-volume transport on an unstructured triangle mesh.

One cell average per triangle per momentum channel, ``u: (K, Nk)``. MUSCL
reconstruction (least-squares cell gradient, Venkatakrishnan limited) feeds a
scalar per-channel upwind flux ``F_c = (v_c.n) u_upwind`` -- each delta-k channel
streams with its own Fermi velocity, so the upwind side of each edge is fixed by
sign(v_c.n). Walls/contacts supply the exterior trace via the (reused)
FermiSurface reflector/contactor; collisions come from the material. Time
stepping is plain RK2 (see _time_evolution); no positivity limiter.

Boundary conditions: reflective walls, fixed-voltage/drift contacts, floating
(zero-current) probes and current sources (per-step scalar level solve), and
periodic faces paired through the mesh lattice. A METIS spatial decomposition
splits cells across the ``r`` comm with a thin halo exchange (FermiSurface
couples k-channels, so k is never split); see :class:`SpatialDecomp` below.

All velocity-independent per-edge weights are precomputed once, so a step is just
one limited-reconstruction pass, two gathers, and three scatters.
"""

from __future__ import annotations
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from qimpy import rc, TreeNode
from qimpy.rc import MPI
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.mpi import ProcessGrid
from ..material import Material
from . import TensorList, Geometry
from ._mesh import load_mesh

_FACE = np.array([[0, 1], [1, 2], [2, 0]])   # local vertex pairs of the 3 faces (CCW)


@dataclass
class FVGeom:
    """Static FV geometry from a triangle (2D) or line (1D) mesh; hot-path arrays
    are torch tensors.  ``area`` is the cell measure (triangle area / interval
    length) and ``elen``/``blen`` the face measure (edge length / 1 for a 1D
    point face).

    Faces split into interior (shared by cells ``eL``/``eR``, normal ``en`` points
    out of ``eL``) and boundary (cell ``bcell``, outward normal ``bn``, ``bmark``
    names the wall/contact). ``eLF``/``eRF``/``bF`` are flat ``cell*n_face +
    localface`` indices (``n_face`` = 3 triangles, 2 line cells) used to gather
    the reconstructed face value of the adjacent cell.
    Periodic faces are paired through the lattice and stored as interior edges
    (the streaming neighbour is the periodic image).
    """

    area: torch.Tensor; inv_area: torch.Tensor; inradius: torch.Tensor   # (K,)
    centroid_np: np.ndarray; vertices_np: np.ndarray; triangles_np: np.ndarray
    face_mid_np: np.ndarray    # (K, n_face, 2) per-cell face (edge) midpoints -> staggered output
    eL: torch.Tensor; eR: torch.Tensor; eLF: torch.Tensor; eRF: torch.Tensor  # (Ne,)
    en: torch.Tensor; elen: torch.Tensor                                  # (Ne,2),(Ne,)
    bcell: torch.Tensor; bF: torch.Tensor; bmark: torch.Tensor            # (Nb,)
    bn: torch.Tensor; blen: torch.Tensor                                  # (Nb,2),(Nb,)
    marker_names: list
    nbr: torch.Tensor          # (K, Nmax) vertex-neighbor cells (self-padded)
    recon: torch.Tensor        # (K, 3, Nmax) face-increment op: d_face = recon @ (u_nbr - u)


def build_fv_geom(mesh, *, dtype: torch.dtype = torch.float64) -> FVGeom:
    """Build the FV geometry from a loaded mesh (``_mesh.MeshResult``).

    Dispatches on cell type: 3 vertices/cell -> 2D triangles, 2 vertices/cell ->
    a 1D line mesh (interval cells; see :func:`_build_fv_geom_1d`).
    """
    tri = np.asarray(mesh.EToV, dtype=int)
    if tri.shape[1] == 2:
        return _build_fv_geom_1d(mesh, dtype=dtype)
    V = np.stack([mesh.VX, mesh.VY], axis=1).astype(float)
    K = len(tri)
    p = V[tri]                                                # (K, 3, 2)
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    area = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    if np.any(area <= 0.0):
        raise ValueError("triangle mesh must be CCW with positive area")
    centroid = p.mean(axis=1)

    fa, fb = _FACE[:, 0], _FACE[:, 1]
    Pa, Pb = p[:, fa], p[:, fb]
    fmid = 0.5 * (Pa + Pb)
    tvec = Pb - Pa
    flen = np.linalg.norm(tvec, axis=2)
    fnrm = np.stack([tvec[..., 1], -tvec[..., 0]], axis=2) / flen[..., None]  # outward
    inradius = area / (0.5 * flen.sum(axis=1))

    # Deduplicate physical edges -> interior (two cells) / boundary (one).
    edge_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for k in range(K):
        for f in range(3):
            key = (min(int(tri[k, fa[f]]), int(tri[k, fb[f]])),
                   max(int(tri[k, fa[f]]), int(tri[k, fb[f]])))
            edge_map.setdefault(key, []).append((k, f))
    interior, boundary = [], []
    for key, hits in edge_map.items():
        if len(hits) == 2:
            (kL, fL), (kR, fR) = hits
            interior.append((kL, fL, kR, fR))
        else:
            (k, f), = hits
            boundary.append((k, f, mesh.edge_marker.get(key, 0)))
    interior = np.array(interior, int).reshape(-1, 4)
    boundary = np.array(boundary, int).reshape(-1, 3)

    # Periodic faces: pair leftover boundary edges across each lattice vector and
    # promote them to interior edges (streaming neighbour = periodic image). Match
    # face midpoints with a KD-tree for robustness on distorted/irregular meshes.
    lattice = getattr(mesh, "_lattice", None)
    if lattice is not None and len(boundary):
        from scipy.spatial import cKDTree
        bmid = fmid[boundary[:, 0], boundary[:, 1]]           # (Nb, 2)
        tol = 1e-6 * float(max(flen.max(), 1.0))
        tree = cKDTree(bmid)
        used = np.zeros(len(boundary), bool)
        paired = []
        for L in np.atleast_2d(np.asarray(lattice, float)):
            dist, j = tree.query(bmid + L, distance_upper_bound=tol)
            for i, (di, ji) in enumerate(zip(dist, j)):
                if di <= tol and ji < len(bmid) and i != ji \
                        and not used[i] and not used[ji]:
                    used[i] = used[ji] = True
                    paired.append((boundary[i, 0], boundary[i, 1],
                                   boundary[ji, 0], boundary[ji, 1]))
        if paired:
            interior = np.vstack([interior, np.array(paired, int)])
            boundary = boundary[~used]

    kL, fL, kR, fR = (interior.T if len(interior) else (np.empty(0, int),) * 4)
    bk, bf, bmark = (boundary.T if len(boundary) else (np.empty(0, int),) * 3)

    # Reconstruction stencil: vertex-neighbors (every cell sharing a vertex), which
    # stays full-rank and well-conditioned on distorted/irregular/boundary cells
    # where the 3 face-neighbors alone are too few or near-collinear.
    v2c: dict[int, list[int]] = {}
    for k in range(K):
        for vtx in tri[k]:
            v2c.setdefault(int(vtx), []).append(k)
    vnbr = [sorted({c for vtx in tri[k] for c in v2c[int(vtx)]} - {k}) for k in range(K)]
    Nmax = max((len(s) for s in vnbr), default=1)
    nbr = np.arange(K)[:, None].repeat(Nmax, axis=1)         # pad slots with self
    # Inverse-distance-weighted least-squares gradient operator per cell:
    #   grad_i = (D^T W D)^{-1} D^T W (u_nbr - u_i),  w_j = 1 / |c_j - c_i|^2.
    # pinv handles any residual rank-deficiency gracefully (min-norm gradient).
    grad_op = np.zeros((K, 2, Nmax))
    for i, js in enumerate(vnbr):
        if len(js) < 2:
            continue
        nbr[i, :len(js)] = js
        D = centroid[js] - centroid[i]                       # (n, 2)
        w = 1.0 / np.maximum((D ** 2).sum(1), 1e-300)        # inverse-distance^2
        sw = np.sqrt(w)
        grad_op[i, :, :len(js)] = np.linalg.pinv(sw[:, None] * D) * sw[None, :]
    # Fuse gradient + centroid->face offsets so a step reconstructs face
    # increments with one (3 x Nmax) @ (Nmax x Nk) matmul per cell.
    face_off = fmid - centroid[:, None]                      # (K, 3, 2)
    recon = np.einsum("kfx,kxg->kfg", face_off, grad_op)     # (K, 3, Nmax)

    def t(a, long=False):
        return torch.tensor(np.ascontiguousarray(a), device=rc.device,
                            dtype=torch.long if long else dtype)

    area_t = t(area)
    return FVGeom(
        area=area_t, inv_area=1.0 / area_t, inradius=t(inradius),
        centroid_np=centroid, vertices_np=V, triangles_np=tri, face_mid_np=fmid,
        eL=t(kL, long=True), eR=t(kR, long=True),
        eLF=t(kL * 3 + fL, long=True), eRF=t(kR * 3 + fR, long=True),
        en=t(fnrm[kL, fL]), elen=t(flen[kL, fL]),
        bcell=t(bk, long=True), bF=t(bk * 3 + bf, long=True), bmark=t(bmark, long=True),
        bn=t(fnrm[bk, bf]), blen=t(flen[bk, bf]), marker_names=list(mesh.marker_names),
        nbr=t(nbr, long=True), recon=t(recon),
    )


def _build_fv_geom_1d(mesh, *, dtype: torch.dtype = torch.float64) -> FVGeom:
    """Build the FV geometry for a 1D line mesh: interval cells on a line.

    Each cell is an interval with two endpoint "faces" (local face 0 = left
    vertex, 1 = right vertex).  The cell measure is its length L (the FV update
    divides by it, so ``area``:=L), the outward face normals are +/- x (unit), and
    point faces have unit measure (``elen``/``blen``:=1, the 1D divergence
    theorem).  The reconstruction reuses the same inverse-distance least-squares
    gradient as 2D; on a line the centroid offsets are purely x, so ``pinv``
    returns the x-gradient and a zero y-gradient (min-norm).  Adjacency is by
    shared vertex: a vertex in two cells is an interior face, in one a boundary
    face (the domain ends), whose marker is looked up as ``(v, v)``.  The material
    is untouched -- velocities stay 2D; only ``v_x = v.n`` streams along the line.
    """
    V = np.stack([mesh.VX, mesh.VY], axis=1).astype(float)    # (Nv, 2), VY ~ 0
    seg = np.asarray(mesh.EToV, dtype=int)                    # (K, 2): [v_left, v_right]
    K = len(seg)
    p = V[seg]                                                # (K, 2, 2): endpoints
    centroid = p.mean(axis=1)                                 # (K, 2)
    L = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)             # (K,) cell length
    if np.any(L <= 0.0):
        raise ValueError("1D line mesh has a zero-length cell")
    area = L                                                  # FV cell measure
    inradius = L                                              # dt = cfl * L / vmax
    fmid = p                                                  # face = the endpoint vertex
    face_off = fmid - centroid[:, None]                       # (K, 2, 2) centroid->face
    fnrm = face_off / np.linalg.norm(face_off, axis=2, keepdims=True)  # +/- x unit normal
    flen = np.ones((K, 2))                                    # point face: unit measure

    # Interior / boundary by shared-vertex dedup.
    vmap: dict[int, list[tuple[int, int]]] = {}
    for k in range(K):
        for f in range(2):
            vmap.setdefault(int(seg[k, f]), []).append((k, f))
    interior, boundary = [], []
    for v, hits in vmap.items():
        if len(hits) == 2:
            (kL, fL), (kR, fR) = hits
            interior.append((kL, fL, kR, fR))
        else:
            (k, f), = hits
            boundary.append((k, f, mesh.edge_marker.get((v, v), 0)))
    interior = np.array(interior, int).reshape(-1, 4)
    boundary = np.array(boundary, int).reshape(-1, 3)
    kL, fL, kR, fR = (interior.T if len(interior) else (np.empty(0, int),) * 4)
    bk, bf, bmark = (boundary.T if len(boundary) else (np.empty(0, int),) * 3)

    # Inverse-distance least-squares gradient over shared-vertex neighbors.
    v2c: dict[int, list[int]] = {}
    for k in range(K):
        for vtx in seg[k]:
            v2c.setdefault(int(vtx), []).append(k)
    vnbr = [sorted({c for vtx in seg[k] for c in v2c[int(vtx)]} - {k}) for k in range(K)]
    Nmax = max((len(s) for s in vnbr), default=1)
    nbr = np.arange(K)[:, None].repeat(Nmax, axis=1)
    grad_op = np.zeros((K, 2, Nmax))
    for i, js in enumerate(vnbr):
        if not js:
            continue
        nbr[i, :len(js)] = js
        D = centroid[js] - centroid[i]                        # (n, 2), y ~ 0
        w = 1.0 / np.maximum((D ** 2).sum(1), 1e-300)
        sw = np.sqrt(w)
        grad_op[i, :, :len(js)] = np.linalg.pinv(sw[:, None] * D) * sw[None, :]
    recon = np.einsum("kfx,kxg->kfg", face_off, grad_op)      # (K, 2, Nmax)

    def t(a, long=False):
        return torch.tensor(np.ascontiguousarray(a), device=rc.device,
                            dtype=torch.long if long else dtype)

    area_t = t(area)
    return FVGeom(
        area=area_t, inv_area=1.0 / area_t, inradius=t(inradius),
        centroid_np=centroid, vertices_np=V, triangles_np=seg, face_mid_np=fmid,
        eL=t(kL, long=True), eR=t(kR, long=True),
        eLF=t(kL * 2 + fL, long=True), eRF=t(kR * 2 + fR, long=True),
        en=t(fnrm[kL, fL]), elen=t(flen[kL, fL]),
        bcell=t(bk, long=True), bF=t(bk * 2 + bf, long=True), bmark=t(bmark, long=True),
        bn=t(fnrm[bk, bf]), blen=t(flen[bk, bf]), marker_names=list(mesh.marker_names),
        nbr=t(nbr, long=True), recon=t(recon),
    )


# --------------------------------------------------------------------------- #
#  Spatial domain decomposition: each rank owns a contiguous block of cells and
#  does O(local) work per step, exchanging only a thin 2-ring halo of ghost-cell
#  averages. Cells are partitioned by METIS (min-cut on the face-neighbour dual
#  graph) and renumbered so every rank's block -- and its checkpoint slice -- is
#  contiguous. The MUSCL stencil needs two rings (a cell's face value uses its
#  1-ring gradient; the flux on its face also uses the neighbour's reconstructed
#  value, hence the neighbour's 1-ring), so the halo is the 2-ring vertex closure.
# --------------------------------------------------------------------------- #
def _dual_graph(EToV) -> list[list[int]]:
    """Face-neighbour adjacency (the FV dual graph): cells sharing an edge."""
    e2c: dict[tuple[int, int], list[int]] = defaultdict(list)
    for k, tri in enumerate(EToV):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            e2c[tuple(sorted((int(tri[a]), int(tri[b]))))].append(k)
    nbr: list[set[int]] = [set() for _ in range(len(EToV))]
    for cells in e2c.values():
        if len(cells) == 2:
            i, j = cells
            nbr[i].add(j)
            nbr[j].add(i)
    return [sorted(s) for s in nbr]


def _coordinate_part(mesh, nparts: int) -> np.ndarray:
    """Fallback partition (no METIS): sort cells along the longer axis into
    equal-count blocks. Correct but with poorer locality on branchy meshes."""
    V = np.stack([mesh.VX, mesh.VY], axis=1)
    cen = V[np.asarray(mesh.EToV, int)].mean(axis=1)
    axis = 0 if np.ptp(cen[:, 0]) >= np.ptp(cen[:, 1]) else 1  # NumPy 2.0: ndarray.ptp removed
    order = np.argsort(cen[:, axis], kind="stable")
    part = np.empty(len(order), np.int32)
    part[order] = np.minimum((np.arange(len(order)) * nparts) // len(order), nparts - 1)
    return part


def partition(mesh, comm: MPI.Comm) -> tuple[np.ndarray, np.ndarray]:
    """Renumber cells into contiguous per-rank blocks.

    Returns ``(perm, bounds)``: applying ``EToV[perm]`` places rank ``r``'s cells
    in the contiguous slice ``[bounds[r], bounds[r+1])``. The partition is a METIS
    min-cut of the face-neighbour dual graph, computed on the head and broadcast
    so every rank agrees exactly; falls back to a coordinate sort if pymetis is
    not installed.
    """
    K = len(mesh.EToV)
    nparts = comm.size
    if nparts == 1:
        return np.arange(K), np.array([0, K], int)
    part = None
    if comm.rank == 0:
        try:
            import pymetis
            _, p = pymetis.part_graph(nparts, adjacency=_dual_graph(mesh.EToV))
            part = np.asarray(p, np.int32)
        except ImportError:
            part = _coordinate_part(mesh, nparts)
    part = comm.bcast(part, root=0)
    perm = np.argsort(part, kind="stable")              # group cells by rank
    bounds = np.concatenate([[0], np.cumsum(np.bincount(part, minlength=nparts))])
    return perm, bounds.astype(int)


class SpatialDecomp:
    """Owned/ghost bookkeeping and halo exchange over a renumbered cell mesh.

    Construct after the cells have been renumbered by :func:`partition` and the
    geometry built, passing the vertex-neighbour table ``nbr`` (global, in the
    renumbered order) and the per-rank ``bounds``. Owned cells of rank ``r`` are
    the contiguous slice ``[bounds[r], bounds[r+1])``.
    """

    def __init__(self, nbr_np: np.ndarray, bounds: np.ndarray,
                 comm: MPI.Comm) -> None:
        self.comm = comm
        self.size = comm.size
        self.rank = comm.rank
        self.K = nbr_np.shape[0]
        self.offset = np.asarray(bounds, int)
        self.start = int(self.offset[self.rank])
        self.stop = int(self.offset[self.rank + 1])
        self.owned = np.arange(self.start, self.stop)

        def ring(cells):
            """1-ring vertex closure of a set of cells (cells + their neighbours)."""
            if not cells:
                return set()
            return set(cells) | set(
                nbr_np[np.asarray(sorted(cells), int)].ravel().tolist())

        own = set(self.owned.tolist())
        ring1 = ring(own)                       # cells to reconstruct (owned + 1-ring)
        ring2 = ring(ring1)                     # cells whose u must be current (2-ring)
        self.recon_rows = np.array(sorted(ring1), int)
        ghosts = np.array(sorted(ring2 - own), int)

        # Halo plans: receive each ghost from its owning rank; send the owned
        # cells that another rank needs (its 2-ring minus its own block).
        self.recv: dict[int, np.ndarray] = {}
        for q in range(self.size):
            sel = ghosts[(ghosts >= self.offset[q]) & (ghosts < self.offset[q + 1])]
            if len(sel) and q != self.rank:
                self.recv[q] = sel
        self.send: dict[int, np.ndarray] = {}
        for q in range(self.size):
            if q == self.rank:
                continue
            owned_q = set(range(int(self.offset[q]), int(self.offset[q + 1])))
            need_q = ring(ring(owned_q)) - owned_q
            sel = self.owned[np.isin(self.owned, np.fromiter(need_q, int, len(need_q)))]
            if len(sel):
                self.send[q] = sel

    def exchange(self, u: torch.Tensor) -> None:
        """Fill this rank's ghost rows of ``u`` (K, Nk) with their owners' values."""
        if not self.recv and not self.send:
            return
        un = u.detach().to("cpu").numpy()
        reqs = []
        recv_bufs = {}
        for q, idx in self.recv.items():
            buf = np.empty((len(idx), un.shape[1]), un.dtype)
            recv_bufs[q] = (buf, idx)
            reqs.append(self.comm.Irecv(buf, source=q, tag=11))
        send_bufs = []
        for q, idx in self.send.items():
            sb = np.ascontiguousarray(un[idx])
            send_bufs.append(sb)
            reqs.append(self.comm.Isend(sb, dest=q, tag=11))
        MPI.Request.Waitall(reqs)
        for q, (buf, idx) in recv_bufs.items():
            u[torch.as_tensor(idx, device=u.device)] = torch.as_tensor(
                buf, device=u.device, dtype=u.dtype)


@dataclass
class _Contact:
    """One boundary contact. ``fixed`` holds a prescribed ghost; a feedback
    contact (``floating`` probe or ``current`` source) solves a scalar level
    (the reservoir chemical potential mu_tilde) each evaluation so its net
    current hits ``target`` (0 for floating).  The reservoir is a genuine
    isotropic Fermi-Dirac  f_c[r] = sigmoid(mu_tilde - xi_r)."""

    name: str
    idx: torch.Tensor                 # boundary-edge indices of this contact
    cur: torch.Tensor                 # (Nsel, Nk) outward number-flux operator
    kind: str = "fixed"
    ghost: Optional[torch.Tensor] = None       # fixed: prescribed exterior trace
    cur_out: Optional[torch.Tensor] = None     # feedback: outflow-only flux op
    target: float = 0.0                        # feedback: desired net current
    level: float = 0.0                         # feedback: last solved mu_tilde
    Br: Optional[torch.Tensor] = None          # feedback: per-radial inflow current cap
    xi_r: Optional[torch.Tensor] = None        # feedback: radial energy nodes (Nr,)
    N_theta: int = 0                           # feedback: angular node count


class FiniteVolume(Geometry):
    """Cell-centered finite-volume geometry on an external triangle mesh."""

    def __init__(
        self,
        *,
        material: Material,
        mesh_file: str,
        contacts: dict[str, Optional[dict]],
        cfl: float = 0.4,
        vk_eps2: float = 0.0,
        compile: bool = False,
        save_rho: bool = False,
        save_terms: bool = False,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Parameters
        ----------
        mesh_file
            :yaml:`Path to an external triangle mesh (.npz) to solve on.`
        contacts
            :yaml:`Dictionary of contact names to parameters (match mesh markers).`
            Each value selects the contact kind: ``{dmu, vD}`` a fixed
            voltage/drift source, ``{floating: true}`` a zero-current probe, and
            ``{I_set: <current>}`` a current source (with optional ``vD``).
        cfl
            :yaml:`CFL number for the explicit step (dt = cfl * inradius / vmax).`
        vk_eps2
            :yaml:`Venkatakrishnan threshold in field^2 units (0 = pure smooth limiter).`
            Set to ~(mesh-scale variation)^2 to stop limiting smooth/low-amplitude
            data and recover strict linearity preservation.
        compile
            :yaml:`torch.compile the limited reconstruction (fuses the per-step
            limiter kernels).` ~3x faster steps on GPU at the cost of a one-time
            compile; leave off for short runs and the test suite.
        """
        TreeNode.__init__(self)
        self.material = material
        self.comm = process_grid.get_comm("r")
        self.mesh_file = mesh_file
        self.contacts = contacts
        self.save_rho = save_rho
        self.save_terms = save_terms
        self._vk_eps2 = float(vk_eps2)

        self.mesh = load_mesh(mesh_file)
        self._mpi = self.comm.size > 1
        if self._mpi:
            # METIS min-cut partition, renumbered so each rank owns a contiguous
            # block (compact halos + direct checkpoint slices). Keep the
            # permutation so the renumbered solution maps back to the input order.
            self._perm, bounds = partition(self.mesh, self.comm)
            self.mesh.EToV = np.asarray(self.mesh.EToV, int)[self._perm]
        else:
            self._perm, bounds = None, None
        g = build_fv_geom(self.mesh, dtype=material.transport_velocity.dtype)
        self.geom = g
        v = material.transport_velocity                       # (Nk, 2)
        self.Nk = v.shape[0]
        self.K = int(g.area.shape[0])
        self._nf = int(g.recon.shape[1])                      # faces/cell: 3 (tri) or 2 (1D)

        # Spatial decomposition: owned cell block, reconstruction rows (owned +
        # 1-ring), owned-incident edges and the halo exchange (see SpatialDecomp).
        if self._mpi:
            self._decomp = SpatialDecomp(g.nbr.detach().cpu().numpy(), bounds, self.comm)
            self._own_start, self._own_stop = self._decomp.start, self._decomp.stop
            self._R = torch.as_tensor(self._decomp.recon_rows,
                                      device=rc.device, dtype=torch.long)
            self._owned_mask = torch.zeros(self.K, 1, dtype=torch.bool, device=rc.device)
            self._owned_mask[self._own_start:self._own_stop] = True
        else:
            self._decomp = None
            self._own_start, self._own_stop = 0, self.K
            self._R = None
            self._owned_mask = None

        # Precompute velocity-weighted, area-scaled edge operators (constant):
        #   into eL: -a*elen/area_L,  into eR: +a*elen/area_R,  a = v_c.n
        a_int = g.en @ v.t()                                  # (Ne, Nk)
        self._maskL = a_int > 0
        self._wL = -a_int * (g.elen * g.inv_area[g.eL])[:, None]
        self._wR = a_int * (g.elen * g.inv_area[g.eR])[:, None]
        self._a_bnd = g.bn @ v.t()                            # (Nb, Nk)
        self._maskB = self._a_bnd > 0
        self._wB = -self._a_bnd * (g.blen * g.inv_area[g.bcell])[:, None]
        # Per-channel density weight; outward number-flux operator per boundary edge.
        self._ncoef = material.get_observables(0.0)[0]        # (Nk,)
        self._cur_b = (g.blen[:, None] * self._ncoef[None, :]) * self._a_bnd  # (Nb,Nk)
        # Restrict per-step work to cells/edges this rank owns (all of them serially).
        lo, hi = self._own_start, self._own_stop
        if self._mpi:
            own_e = ((g.eL >= lo) & (g.eL < hi)) | ((g.eR >= lo) & (g.eR < hi))
            self._eloc = torch.where(own_e)[0]
            self._bloc = torch.where((g.bcell >= lo) & (g.bcell < hi))[0]
        else:
            self._eloc = None
            self._bloc = None
        self._setup_boundary(material)

        # De-aliasing projector.  When the material's angular quadrature oversamples
        # its modes -- FermiSurface uses an even N_theta > 2M+1 so the velocity set
        # is mirror-symmetric (left-right contact symmetry) -- there are nodal DOFs
        # carrying no represented harmonic.  The per-channel MUSCL limiter is
        # nonlinear and excites them; the modal collision operator cannot damp them,
        # so under strong collisions they grow without bound.  Projecting the state
        # onto the represented modes each evaluation removes that aliased content;
        # it is exactly the identity on n, the current, and every retained harmonic.
        # No-op for square transforms or materials without modal transforms
        # (single_band, ab_initio).  Applied as  u @ self._proj.
        # The unrepresented ("ghost") subspace has tiny rank r = N_theta - (2M+1)
        # (1-3), so we remove it with a rank-r update  u -= (u @ A) @ B^T  rather
        # than a dense Nk*Nk matmul -- numerically identical, ~Nk/r times cheaper.
        # Always on: the ghost lies in ker(to_modes) (collision cannot damp it) yet
        # the nonlinear limiter excites it, so it must be projected out each step.
        self._dl_A = self._dl_B = None
        if hasattr(material, "to_modes") and hasattr(material, "from_modes"):
            eye = torch.eye(self.Nk, device=rc.device, dtype=v.dtype)
            proj = material.from_modes(material.to_modes(eye))   # (Nk, Nk) projector
            ghost = eye - proj                                   # onto unrepresented DOFs
            if ghost.abs().max() > 1e-10:                        # only if oversampled
                U, S, Vh = torch.linalg.svd(ghost)
                # Numerical-rank cutoff scaled to the working precision. An
                # absolute threshold (1e-8) sits below the fp32 modal round-trip
                # floor (~1e-6), so in fp32 it counts ~Nk noise singular values as
                # ghost directions. sqrt(eps) lands safely between the noise floor
                # and the O(1) genuine ghost in both fp32 and fp64 (fp64: ~1.5e-8,
                # matching the old cutoff; fp32: ~3.4e-4).
                tol = float(S[0]) * torch.finfo(v.dtype).eps ** 0.5
                r = int((S > tol).sum())
                self._dl_A = (U[:, :r] * S[:r]).contiguous()     # (Nk, r)
                self._dl_B = Vh[:r].T.contiguous()               # (Nk, r)

        # Ballistic short-circuit: when the material's collision+field operator is
        # identically zero (rates and cyclotron speed both zero) its rho_dot is a
        # no-op, so skip the per-step call entirely -- it otherwise allocates a
        # zero tensor and (via the rates check) forces a GPU->CPU sync each step.
        rm = getattr(material, "rates_modal", None)
        ks = getattr(material, "k_speed", 0.0)
        has_ee = hasattr(material, "ee_scattering")   # microscopic e-e: NOT in rates_modal
        self._skip_collision = bool(
            rm is not None and float(rm.abs().sum()) == 0.0 and float(ks) == 0.0
            and not has_ee
        )

        # Optionally fuse the per-step kernels with torch.compile.  Serial: compile
        # the whole spatial RHS (reconstruction + limiter + flux scatter + boundary)
        # so the flux glue fuses into the reconstruction graph and the kernel-launch
        # overhead collapses.  MPI: shapes are dynamic (owned-edge gathers), so only
        # the reconstruction is compiled.
        self._faces_fn = self._faces
        self._srhs_fn = self._spatial_rhs
        if compile:
            # This workload is kernel-launch- and bandwidth-bound (a long chain of
            # small elementwise ops). Benchmarked on a T4: "max-autotune" (kernel
            # fusion + tuning) is fastest; "reduce-overhead" (CUDA graphs) is a touch
            # slower here because the RK2 stages feed changing inputs, so the
            # cudagraph input copies offset the launch-overhead savings. Override
            # with QIMPY_COMPILE_MODE if a workload benefits from a different mode.
            mode = os.environ.get("QIMPY_COMPILE_MODE", "max-autotune")
            try:
                if self._mpi:
                    self._faces_fn = torch.compile(self._faces, mode=mode)
                else:
                    self._srhs_fn = torch.compile(self._spatial_rhs, mode=mode)
            except Exception:               # older torch / no backend -> eager
                self._faces_fn, self._srhs_fn = self._faces, self._spatial_rhs

        vmax = float(v.norm(dim=1).max())
        dt_local = float(cfl) * float(g.inradius.min()) / max(vmax, 1e-300)
        self.dt_max = self.comm.allreduce(dt_local, op=MPI.MIN)

        rho0 = getattr(material, "rho0", None)
        if rho0 is not None:
            self._u = rho0.flatten().to(rc.device, v.dtype)[None, :].repeat(self.K, 1)
        else:
            self._u = torch.zeros(self.K, self.Nk, device=rc.device, dtype=v.dtype)

        # ---- moving drift-frame (exact scheme) setup ---------------------------
        # Each cell carries a frame (mu,Te,u); the conserved lab densities U=(n,Jx,
        # Jy,E) are the primary object, f the shape.  The per-edge streaming velocity
        # is (v_node(frame)+u).n, so the precomputed global a_int/_wL/_wR are kept
        # only for the ballistic fallback (material.moving_frame=False).
        self._moving = bool(getattr(material, "moving_frame", False))
        self._cfl = float(cfl)
        if self._moving:
            fs = material
            cth = torch.cos(fs.angular.theta); sth = torch.sin(fs.angular.theta)   # (Nθ,)
            # frame-INDEPENDENT geometric factor  k̂·n̂  per (edge, angular node)
            self._ang_int = g.en[:, 0:1] * cth[None, :] + g.en[:, 1:2] * sth[None, :]   # (Ne,Nθ)
            self._ang_bnd = g.bn[:, 0:1] * cth[None, :] + g.bn[:, 1:2] * sth[None, :]   # (Nb,Nθ)
            self._xi = fs.radial.xi                                                     # (Nr,)
            # seed U from the uniform-equilibrium frame (f already = rho0 isotropic)
            mu0 = torch.full((self.K,), fs.E_F, device=rc.device, dtype=v.dtype)
            Te0 = torch.full((self.K,), fs.T_temp, device=rc.device, dtype=v.dtype)
            u0 = torch.zeros(self.K, 2, device=rc.device, dtype=v.dtype)
            self._U = fs.U_from_frame(mu0, Te0, u0)            # (K,4)
            self._Te = Te0                                     # frame-recovery warm start

        self._stash_t, self._stash_i, self._stash_obs = [], [], []
        self._stash_terms = []   # per-frame (4, K_own, Nr*dim): [a, lin, quad, cub]
        self._stash_edge = []    # per-frame (Nedge, 2) face normal-flux [j.n^, q.n^]
        self._edge_geom_cache = None   # static (midpoints, normals, lengths), lazy

    def _setup_boundary(self, material: Material) -> None:
        """Group boundary edges by marker into a wall (reflector) set and one
        contact object per parametrized marker; unparametrized markers reflect.

        Under decomposition each rank keeps only the boundary edges on its owned
        cells; a feedback contact's capacity (``den``/``base``) is summed across
        ranks so every rank solves the same global level."""
        g = self.geom
        names = g.marker_names
        name_of = [names[m] if 0 <= m < len(names) else "wall"
                   for m in g.bmark.tolist()]
        lo, hi = self._own_start, self._own_stop
        bcell = g.bcell.tolist()
        owns = [lo <= bcell[i] < hi for i in range(len(name_of))]  # this rank's edges
        is_c = lambda nm: (nm in self.contacts) and (self.contacts[nm] is not None)
        wall = np.array([i for i, nm in enumerate(name_of)
                         if owns[i] and not is_c(nm)], int)
        self._wall = torch.as_tensor(wall, device=rc.device, dtype=torch.long)
        self._reflector = material.get_reflector(g.bn[self._wall]) if wall.size else None
        # Per-(boundary-edge) wall/contact classification for the moving-frame KFVS
        # boundary: True = reflecting wall, False = contact.  Selects the ghost
        # equilibrium drift (a specular wall flips the normal drift so the reflected
        # core cancels the incident; a contact keeps the cell drift, its reservoir
        # entering only through the shell deviation).  Covers ALL boundary edges.
        self._is_wall_b = torch.as_tensor(
            [not is_c(nm) for nm in name_of], device=rc.device, dtype=torch.bool)
        # The full-f reflector's diffuse re-emit is NONLINEAR in the wall trace
        # (a genuine Fermi-Dirac at a per-face chemical potential), so it is
        # evaluated per step rather than collapsed into a precomputed per-edge
        # matrix (as a purely linear reflector could be).

        def allreduce(x: float) -> float:
            return self.comm.allreduce(x) if self._mpi else x

        self._contacts: list[_Contact] = []
        for nm, params in self.contacts.items():
            if params is None:
                continue
            sel = np.array([i for i, x in enumerate(name_of)
                            if x == nm and owns[i]], int)
            ci = torch.as_tensor(sel, device=rc.device, dtype=torch.long)
            cur = self._cur_b[ci]                             # (Nsel, Nk) (maybe empty)
            params = dict(params)
            floating = bool(params.pop("floating", False))
            i_set = params.pop("I_set", None)
            if floating or (i_set is not None):
                # Full-f current/floating source: the reservoir is a genuine
                # isotropic Fermi-Dirac  f_c[r] = sigmoid(mu_tilde - xi_r).  The
                # net emitted current is nonlinear in mu_tilde, so the geometry
                # layer solves a 1-D Newton for mu_tilde each step (see _exterior).
                # Precompute the per-radial inflow current capacity
                #   B_r = sum_{faces, ordinates in inflow} cur[.,r,.]   (< 0)
                # and keep cur_out for the outflow (interior) current term.
                Nr = int(material.Nr)
                Nth = int(material.angular.N_theta)
                cur_in = torch.where(self._a_bnd[ci] < 0, cur, torch.zeros_like(cur))
                cur_out = torch.where(self._a_bnd[ci] > 0, cur, torch.zeros_like(cur))
                Br_local = cur_in.reshape(-1, Nr, Nth).sum((0, 2))    # (Nr,)
                Br = torch.tensor([allreduce(float(x)) for x in Br_local],
                                  device=rc.device, dtype=cur.dtype)   # (Nr,) global
                self._contacts.append(_Contact(
                    name=nm, idx=ci, cur=cur, kind="current", cur_out=cur_out,
                    target=(0.0 if floating else float(i_set)),
                    Br=Br, xi_r=material.radial.xi.to(cur.dtype),
                    N_theta=Nth))
            else:
                ghost = material.get_contactor(g.bn[ci], **params)(0.0)
                self._contacts.append(_Contact(
                    name=nm, idx=ci, cur=cur, kind="fixed", ghost=ghost))

    def _limited_faces(self, uc: torch.Tensor, un: torch.Tensor,
                       recon: torch.Tensor) -> torch.Tensor:
        """Venkatakrishnan-limited face values for a set of cells.

        ``uc`` (n, Nk) cell averages, ``un`` (n, Nmax, Nk) their vertex-neighbor
        averages, ``recon`` (n, 3, Nmax) the fused gradient->face operator. Smooth
        (differentiable) limiter -> clean steady-state convergence; per face, with
        increment d = u_face-u and same-sign headroom D1 (to the neighbor max/min):
            phi = (D1^2 + 2 D1 d + e) / (D1^2 + D1 d + 2 d^2 + e),  e = vk_eps2,
        capped at 1 (never amplify the LSQ gradient) and min-ed over the 3 faces.
        """
        d = torch.einsum("nfg,ngc->nfc", recon, un - uc[:, None])   # (n, 3, Nk)
        hi = (torch.maximum(uc, un.amax(1)) - uc)[:, None]    # headroom up   (>= 0)
        lo = (torch.minimum(uc, un.amin(1)) - uc)[:, None]    # headroom down (<= 0)
        D1 = torch.where(d >= 0, hi, lo)                      # same sign as d
        e = self._vk_eps2
        # Denominator is D1^2 + D1 d + 2 d^2 + e >= 2 d^2 > 0 for d != 0 in exact
        # arithmetic, but in fp32 it can underflow to a subnormal/zero at locally
        # flat cells (d a denormal-tiny roundoff with D1 == 0), giving 0/0 = NaN.
        # Guard on the denominator being a normal float rather than on d != 0:
        # where it isn't, the cell is flat and the limiter is 1 (no limiting).
        num = D1 * D1 + 2 * D1 * d + e
        den = D1 * D1 + D1 * d + 2 * d * d + e
        phi = torch.where(
            den > torch.finfo(d.dtype).tiny,
            (num / den).clamp(max=1.0),
            torch.ones_like(d),
        ).amin(1)[:, None]                                    # (n, 1, Nk)
        return uc[:, None] + phi * d

    def _faces(self, u: torch.Tensor) -> torch.Tensor:
        """Reconstructed face values, (K, n_face, Nk). Serial reconstructs every
        cell; under decomposition only the rows this rank needs (owned + 1-ring)
        are filled, the rest left zero (their faces are never read)."""
        g = self.geom
        if self._R is None:
            return self._limited_faces(u, u[g.nbr], g.recon)
        R = self._R
        uf = u.new_zeros(self.K, self._nf, self.Nk)
        uf[R] = self._limited_faces(u[R], u[g.nbr[R]], g.recon[R])
        return uf

    def _exterior(self, uMb: torch.Tensor, t: float) -> torch.Tensor:
        """Exterior ghost at boundary edges: reflector on walls, prescribed or
        feedback-solved contactor on contacts. Feedback contacts solve a scalar
        level so the net current equals their target; only inflow channels are
        consumed downstream by the upwind flux."""
        uP = uMb.clone()
        if self._reflector is not None and self._wall.numel():
            # Full-f wall: nonlinear (specular + genuine-FD diffuse) per step.
            uP[self._wall] = self._reflector(uMb[self._wall]).to(uP)
        for c in self._contacts:
            if c.kind == "fixed":
                uP[c.idx] = c.ghost.to(uP)
            else:
                # Full-f reservoir: solve mu_tilde so the net current hits target,
                #   C_out + sum_r sigmoid(mu_tilde - xi_r) * B_r = target,
                # then emit the isotropic Fermi-Dirac f_c[r] = sigmoid(mu-xi_r).
                # C_out (outflow, interior-carried) is summed across ranks so the
                # solved level is global; B_r is already global.
                C_out = float((c.cur_out * uMb[c.idx]).sum())
                if self._mpi:
                    C_out = self.comm.allreduce(C_out)
                Br, xi = c.Br, c.xi_r
                rhs = c.target - C_out
                F0 = torch.clamp(rhs / Br.sum(), 1e-6, 1.0 - 1e-6)  # flat-f init
                mu = torch.log(F0 / (1.0 - F0))
                for _ in range(8):                                  # 1-D Newton
                    fw = torch.sigmoid(mu - xi)                     # (Nr,)
                    g  = (fw * Br).sum() - rhs
                    gp = (fw * (1.0 - fw) * Br).sum()               # < 0
                    mu = mu - g / gp
                c.level = float(mu)
                fw = torch.sigmoid(mu - xi).clamp(0.0, 1.0)         # (Nr,)
                Nsel = int(c.idx.numel())
                # Explicit last dim (not -1): under MPI a rank may own zero of
                # this contact's cells (Nsel=0); reshape(-1) is ambiguous on a
                # 0-element tensor, so give the exact channel count Nr*N_theta.
                uP[c.idx] = (fw[None, :, None]
                             .expand(Nsel, xi.numel(), c.N_theta)
                             .reshape(Nsel, xi.numel() * c.N_theta).to(uP))
        return uP

    def _spatial_rhs(self, u: torch.Tensor, t: float) -> torch.Tensor:
        g = self.geom
        u = self._dealias(u)                                  # de-alias (fused into graph)
        uf = self._faces_fn(u).reshape(-1, self.Nk)           # (K*3, Nk)
        dudt = torch.zeros_like(u)
        e = self._eloc                                        # owned-incident edges
        eL = g.eL if e is None else g.eL[e]
        eR = g.eR if e is None else g.eR[e]
        eLF = g.eLF if e is None else g.eLF[e]
        eRF = g.eRF if e is None else g.eRF[e]
        maskL = self._maskL if e is None else self._maskL[e]
        wL = self._wL if e is None else self._wL[e]
        wR = self._wR if e is None else self._wR[e]
        uup = torch.where(maskL, uf[eLF], uf[eRF])            # interior upwind trace
        dudt.index_add_(0, eL, wL * uup)
        dudt.index_add_(0, eR, wR * uup)
        if g.bcell.numel():
            uMb = uf[g.bF]
            uup_b = torch.where(self._maskB, uMb, self._exterior(uMb, t))
            wbu = self._wB * uup_b
            if self._bloc is None:
                dudt.index_add_(0, g.bcell, wbu)
            else:
                b = self._bloc
                dudt.index_add_(0, g.bcell[b], wbu[b])
        return dudt

    def _dealias(self, u: torch.Tensor) -> torch.Tensor:
        """Project onto the represented angular modes (rank-r ghost removal)."""
        if self._dl_A is None:
            return u
        return u - (u @ self._dl_A) @ self._dl_B.T

    # ==================================================================
    #  Moving drift-frame exact scheme (Stage 1, ballistic).
    #  March conserved U=(n,Jx,Jy,E) by frame-independent lab fluxes with a
    #  kinetic flux-vector-split (KFVS) shared-face flux (single-valued ->
    #  telescopes -> exact continuity at any per-cell frame); transport the shape
    #  f with the per-edge frame velocity; recover the post-march frame; project
    #  the deviation moment-free (moment(f)=U).
    # ==================================================================
    def _edge_speed(self, mu, Te, u, ang, eL, eR, bcell=None):
        """Per-(edge, node) streaming velocity (v_node(frame_e)+u_e).n.
        Interior: face-average the two neighbours' frames; boundary: the cell frame."""
        fs = self.material
        if bcell is None:
            mu_e = 0.5 * (mu[eL] + mu[eR]); Te_e = 0.5 * (Te[eL] + Te[eR])
            u_e = 0.5 * (u[eL] + u[eR]); en = self.geom.en
        else:
            mu_e, Te_e, u_e, en = mu[bcell], Te[bcell], u[bcell], self.geom.bn
        eps = mu_e[:, None] + Te_e[:, None] * self._xi                       # (Nedge,Nr)
        vfac = (fs.hbar / fs.mstar) * torch.sqrt(
            torch.clamp(2.0 * fs.mstar * eps, min=0.0)) / fs.hbar            # (Nedge,Nr)
        udotn = (u_e * en).sum(-1)                                          # (Nedge,)
        return (vfac[:, :, None] * ang[:, None, :]
                + udotn[:, None, None]).reshape(ang.shape[0], self.Nk)      # (Nedge,Nk)

    def _march_U(self, f, mu, Te, u, t, uf):
        """dU/dt from the frame-independent lab fluxes (n u, Pi, q) via a kinetic
        flux-vector-split (KFVS) shared face + boundary booking through the existing
        contactor/reflector ghost.  ``uf`` are the (reused) reconstructed face values of f."""
        fs, g = self.material, self.geom
        eL, eR, en, elen, iA = g.eL, g.eR, g.en, g.elen, g.inv_area
        U = self._U
        dU = torch.zeros_like(U)
        # Staggered output: per-edge normal-flux DENSITIES [number j.n^, energy q.n^],
        # the SAME KFVS lab flux booked below but kept per unit edge length (i.e.
        # pre-`* elen`), so it lives at the face midpoint.  update_stash reads these.
        self._Ff_int_dens = U.new_zeros((g.eL.shape[0], 2))
        self._Ff_bnd_dens = U.new_zeros((g.bcell.shape[0], 2))
        # ---- interior: kinetic flux-vector split (KFVS) -- NO Rusanov ----------
        # One faithful Boltzmann discretization: the U-flux is the exact lab moment
        # of the SAME per-node kinetic upwind that streams f (see _shape_rhs):
        #   Ff = [ 1/2(Phi^eq_L+Phi^eq_R) - 1/2(Psi^eq_R-Psi^eq_L) ]      equilibrium
        #      + cnorm Sum_q wk_e a_q (f_up - f0) w_a,e                    deviation.
        # The EQUILIBRIUM is velocity-split (KFVS-on-f0): the v_F sound modes live in
        # the frame variation, so a central equilibrium carries no dissipation and
        # forward-Euler blows up (verified von-Neumann + 1D).  The kinetic |v.n^|
        # moment Psi^eq supplies the per-characteristic dissipation (PSD, ~0.42-0.85
        # v_F, sharper than Rusanov's blanket v_max).  The DEVIATION is the exact lab
        # moment of the per-node upwind occupation f_up -- identical to the shape
        # stream.  Single-valued face flux booked +- => telescopes exactly; the
        # dissipation vanishes for uniform data => exact lab flux there.
        eLF, eRF = g.eLF, g.eRF
        Nr, Nth = fs.Nr, fs.angular.N_theta
        nx, ny = en[:, 0], en[:, 1]
        mu_e = 0.5 * (mu[eL] + mu[eR]); Te_e = 0.5 * (Te[eL] + Te[eR])
        u_e = 0.5 * (u[eL] + u[eR])
        a_e = self._edge_speed(mu, Te, u, self._ang_int, eL, eR)            # (Ne,Nk) v_lab.n^
        # (1) equilibrium: central drifted-FD flux + KFVS-on-f0 kinetic dissipation
        Ff = (0.5 * (fs.eq_flux(mu[eL], Te[eL], u[eL], nx, ny)
                     + fs.eq_flux(mu[eR], Te[eR], u[eR], nx, ny))
              - 0.5 * (fs.eq_abs_flux(mu[eR], Te[eR], u[eR], nx, ny)
                       - fs.eq_abs_flux(mu[eL], Te[eL], u[eL], nx, ny)))     # (Ne,4)
        # (2) deviation: exact lab moment of the per-node upwind occupation f_up
        f_up = torch.where(a_e > 0, uf[eLF], uf[eRF])                        # (Ne,Nk)
        dfu = (f_up - fs.rho0).reshape(-1, Nr, Nth)                          # core cancels
        Jc = fs.mstar * Te_e / fs.hbar ** 2
        wk = (fs.radial.flat_w[:, None] * fs.angular.wphi) * Jc[:, None, None]
        ab = a_e.reshape(-1, Nr, Nth)
        kb = fs._kbar(mu_e, Te_e)[:, :, None]
        cph, sph = torch.cos(fs.angular.theta), torch.sin(fs.angular.theta)
        kx, ky = kb * cph, kb * sph
        eps = (mu_e[:, None] + Te_e[:, None] * fs.radial.xi)[:, :, None]
        ux, uy = u_e[:, 0][:, None, None], u_e[:, 1][:, None, None]
        base = fs.cnorm * wk * ab * dfu                                      # (Ne,Nr,Nth)
        Ff = Ff + torch.stack(
            [base.sum((-1, -2)),
             (base * (fs.hbar * kx + fs.mstar * ux)).sum((-1, -2)),
             (base * (fs.hbar * ky + fs.mstar * uy)).sum((-1, -2)),
             (base * (eps + fs.hbar * (ux * kx + uy * ky)
                      + 0.5 * fs.mstar * (u_e * u_e).sum(-1)[:, None, None])).sum((-1, -2))], -1)
        self._Ff_int_dens = torch.stack((Ff[:, 0], Ff[:, 3]), dim=-1)        # (Ne,2) j.n^, q.n^
        Ff = Ff * elen[:, None]                                              # (Ne,4)
        dU.index_add_(0, eL, -Ff * iA[eL, None])
        dU.index_add_(0, eR, +Ff * iA[eR, None])
        # boundary: the SAME kinetic flux-vector split as the interior, between the
        # cell frame and a per-edge GHOST frame.  The drifted-FD EQUILIBRIUM (filled
        # core + f0 shell) is split -- NOT booked centrally -- so at a specular wall
        # the reflected core's normal drift cancels the incident (zero net normal
        # mass flux); the shell DEVIATION of the actual ghost (reflector/contactor)
        # is booked upwind.  Wall ghost drift = cell drift with the normal component
        # flipped (u -> u - 2(u.n^)n^); a contact keeps the cell drift (central eq),
        # its reservoir entering through the deviation.  See final_review2 issue 3.
        if g.bcell.numel():
            bn, blen, bc = g.bn, g.blen, g.bcell
            bnx, bny = bn[:, 0], bn[:, 1]
            uMb = uf[g.bF]
            uP = self._exterior(uMb, t)
            u_bc = u[bc]
            un_b = (u_bc * bn).sum(-1)                          # u.n^ (normal drift)
            u_refl = u_bc - 2.0 * un_b[:, None] * bn            # specular: flip normal drift
            u_g = torch.where(self._is_wall_b[:, None], u_refl, u_bc)   # wall / contact
            u_e = 0.5 * (u_bc + u_g)                            # face-average drift
            # (1) equilibrium KFVS split (cell frame vs ghost frame); at a wall the
            # mass channel cancels analytically (eq_flux_n odd, eq_abs_flux_n even in
            # the normal drift), so the ~80%-of-mass filled core no longer leaks.
            EQ = (0.5 * (fs.eq_flux(mu[bc], Te[bc], u_bc, bnx, bny)
                         + fs.eq_flux(mu[bc], Te[bc], u_g, bnx, bny))
                  - 0.5 * (fs.eq_abs_flux(mu[bc], Te[bc], u_g, bnx, bny)
                           - fs.eq_abs_flux(mu[bc], Te[bc], u_bc, bnx, bny)))     # (Nb,4)
            # (2) shell deviation of the actual ghost, upwound by the face-average
            # lab velocity a_b = v_node(frame) + u_e.n^ (the normal drift vanishes at
            # a wall, so the reflector's own mass balance closes the deviation too).
            a_b = self._edge_speed(mu, Te, u, self._ang_bnd, None, None, bcell=bc)
            uedn = (u_e * bn).sum(-1)                           # u_e.n^ (0 at a wall)
            a_b = a_b + (uedn - un_b)[:, None]                  # swap u.n^ -> u_e.n^
            f_bnd = torch.where(a_b > 0, uMb, uP)               # outflow interior / inflow ghost
            Nr, Nth = fs.Nr, fs.angular.N_theta
            dfb = (f_bnd - fs.rho0).reshape(-1, Nr, Nth)
            Jc = fs.mstar * Te[bc] / fs.hbar ** 2
            wk = (fs.radial.flat_w[:, None] * fs.angular.wphi) * Jc[:, None, None]
            ab = a_b.reshape(-1, Nr, Nth)
            kb = fs._kbar(mu[bc], Te[bc])[:, :, None]
            cph, sph = torch.cos(fs.angular.theta), torch.sin(fs.angular.theta)
            kx, ky = kb * cph, kb * sph
            eps = (mu[bc][:, None] + Te[bc][:, None] * fs.radial.xi)[:, :, None]
            ux, uy = u_e[:, 0][:, None, None], u_e[:, 1][:, None, None]
            base = fs.cnorm * wk * ab * dfb
            flux_n = EQ[:, 0] + base.sum((-1, -2))
            flux_J = EQ[:, 1:3] + torch.stack(
                [(base * (fs.hbar * kx + fs.mstar * ux)).sum((-1, -2)),
                 (base * (fs.hbar * ky + fs.mstar * uy)).sum((-1, -2))], -1)
            flux_E = EQ[:, 3] + (base * (eps + fs.hbar * (ux * kx + uy * ky)
                                 + 0.5 * fs.mstar * (u_e ** 2).sum(-1)[:, None, None])).sum((-1, -2))
            self._Ff_bnd_dens = torch.stack((flux_n, flux_E), dim=-1)        # (Nb,2) j.n^, q.n^
            dU[:, 0].index_add_(0, bc, -flux_n * blen * iA[bc])
            dU[:, 1:3].index_add_(0, bc, -flux_J * (blen * iA[bc])[:, None])
            dU[:, 3].index_add_(0, bc, -flux_E * blen * iA[bc])
        # momentum relaxation (tau_p): the DRIFT momentum lives in U, and the shape
        # collision cannot touch it (projection re-pins), so book -J/tau_p here.
        # Elastic: energy is retained (drift KE -> heat via the closure), E untouched.
        tip = float(getattr(fs, "tau_inv_p", 0.0))
        if tip:
            dU[:, 1:3] = dU[:, 1:3] - tip * self._U[:, 1:3]
        return dU

    def _grad(self, q):
        """Green-Gauss per-cell gradient of a FRAME field (distinct from qimpy's
        least-squares MUSCL `recon` for f).  q:(K,) scalar or (K,d) vector ->
        (K,2) or (K,2,d).  grad q = (1/A)[Σ_int ½(q_L+q_R) n̂ ℓ (+eL,−eR) + Σ_bnd q n̂ ℓ];
        boundary face value = owner cell (frame is a per-cell reservoir)."""
        g = self.geom
        vec = (q.dim() == 2)
        out = q.new_zeros((self.K, 2) + ((q.shape[1],) if vec else ()))
        qe = 0.5 * (q[g.eL] + q[g.eR])
        wl = g.elen[:, None] * g.en
        c = wl[:, :, None] * qe[:, None, :] if vec else wl * qe[:, None]
        out.index_add_(0, g.eL, c)
        out.index_add_(0, g.eR, -c)
        if g.bcell.numel():
            qb = q[g.bcell]
            wb = g.blen[:, None] * g.bn
            cb = wb[:, :, None] * qb[:, None, :] if vec else wb * qb[:, None]
            out.index_add_(0, g.bcell, cb)
        return out * g.inv_area.reshape((-1,) + (1,) * (out.dim() - 1))

    def _transportG(self, f, uf, Jz, mu, Te, u, xidot, phidot, glo, ghi, t):
        """dG for a shell density G=f·𝒥: real-space div (mesh, per-edge frame velocity)
        + k-space grid-motion div (14, material.kspace_div).  f=None -> the VOLUME
        field 𝒥 (f≡1, boundary ghost f=1) for the GCL pass.  G_face = a·ℓ·(f𝒥)_up
        with the UPWIND-cell 𝒥 so G and 𝒥 share every face/upwind decision (GCL)."""
        g = self.geom
        eL, eR, eLF, eRF, elen, iA = g.eL, g.eR, g.eLF, g.eRF, g.elen, g.inv_area
        a_e = self._edge_speed(mu, Te, u, self._ang_int, eL, eR)
        upL = a_e > 0
        f_up = torch.where(upL, uf[eLF], uf[eRF]) if f is not None else 1.0
        J_up = torch.where(upL, Jz[eL, None], Jz[eR, None])
        flux = a_e * elen[:, None] * (f_up * J_up)
        dG = torch.zeros(self.K, self.Nk, device=a_e.device, dtype=a_e.dtype)
        dG.index_add_(0, eL, -flux * iA[eL, None])
        dG.index_add_(0, eR, +flux * iA[eR, None])
        if g.bcell.numel():
            bc = g.bcell
            un_b = (u[bc] * g.bn).sum(-1)                        # u.n^
            u_g = torch.where(self._is_wall_b[:, None],
                              u[bc] - 2.0 * un_b[:, None] * g.bn, u[bc])   # wall reflect / contact
            uedn = (0.5 * (u[bc] + u_g) * g.bn).sum(-1)          # u_e.n^ (0 at a wall)
            a_b = self._edge_speed(mu, Te, u, self._ang_bnd, None, None, bcell=bc)
            a_b = a_b + (uedn - un_b)[:, None]                   # same wall swap as _march_U:
            #                                                     shape streams consistently w/ U-flux
            if f is not None:
                uMb = uf[g.bF]; uP = self._exterior(uMb, t)
                f_bnd = torch.where(a_b > 0, uMb, uP)
            else:
                f_bnd = 1.0
            Jb = Jz[bc, None]
            fluxb = a_b * g.blen[:, None] * (f_bnd * Jb)
            dG.index_add_(0, bc, -fluxb * iA[bc, None])
        Gf = (f * Jz[:, None]) if f is not None else Jz[:, None].expand(self.K, self.Nk)
        return dG + self.material.kspace_div(Gf, xidot, phidot, glo, ghi)

    def _frame_adv(self, q, mu, Te, u):
        """(v+u)·∇_r q per (cell, node) in DIVERGENCE form ∇·((v+u)q) − q ∇·(v+u), using
        the transport's OWN per-node upwind face flux -- the same-operator choice, so the
        discrete material derivative that sets ξ̇',φ̇ is consistent with the flux that
        streams f𝒥 (discrete D_mesh k → 0: the moving-mesh shape transport becomes
        free-stream / moment preserving, not just first-order).  q:(K,m) -> (K,m,Nk).
        Zero-gradient at the boundary (interior faces only; the frame is a per-cell
        reservoir), which also drops the spurious boundary term the Green-Gauss ∇ carried."""
        g = self.geom
        eL, eR, elen, iA = g.eL, g.eR, g.elen, g.inv_area
        a_e = self._edge_speed(mu, Te, u, self._ang_int, eL, eR)      # (Ne,Nk) = (v+u)·n̂
        fL = elen[:, None] * a_e
        div1 = q.new_zeros(self.K, self.Nk)                           # ∇·(v+u)
        div1.index_add_(0, eL, fL * iA[eL, None]); div1.index_add_(0, eR, -fL * iA[eR, None])
        up = a_e > 0
        divq = q.new_zeros(self.K, q.shape[1], self.Nk)              # ∇·((v+u)q)
        for j in range(q.shape[1]):
            fq = fL * torch.where(up, q[eL, j, None], q[eR, j, None])
            divq[:, j].index_add_(0, eL, fq * iA[eL, None])
            divq[:, j].index_add_(0, eR, -fq * iA[eR, None])
        return divq - q[:, :, None] * div1[:, None, :]              # (K,m,Nk)

    def _cfl_substep(self, mu, Te, u, xidot, phidot):
        """Largest stable substep from the real-space (HEATED band speed |v_node|+|u|)
        AND k-space (|ξ̇'|/w_ξ, |φ̇|/w_φ) signal speeds -- self-stable at ANY passed dt.
        Uses the heated band speed ħ k̄_max/m* (NOT the fixed base-T v_speed, which
        under-bounds it when the shear self-heats) and the radial CONTROL VOLUME
        flat_w (< node spacing) so neither signal speed is under-estimated."""
        fs = self.material
        vmax = float((fs.hbar / fs.mstar) * fs._kbar(mu, Te).max()) + float(u.norm(dim=1).max())
        real = float(self.geom.inradius.min()) / max(vmax, 1e-30)
        ksp = min(float(fs.radial.flat_w.min()) / max(float(xidot.abs().max()), 1e-30),
                  float(fs.angular.wphi) / max(float(phidot.abs().max()), 1e-30))
        return self._cfl * min(real, ksp)

    def step_moving_frame(self, t: float, dt: float) -> None:
        """Advance by dt with the coupled exact scheme, internally SUBSTEPPING to the
        real-space + k-space CFL so the solver is self-stable at any passed dt (the
        CFL is not a caller/harness responsibility).  Each substep: recover frame ->
        KFVS U-march + BC -> full (14)+(18) shape transport + GCL -> recover post-frame
        -> moment-free projection + Pauli.  All at τ_p=∞."""
        fs = self.material
        left = float(dt); n_sub = 0
        while left > 1e-12 * float(dt) + 1e-300:
            n_sub += 1
            if n_sub > 100000:
                raise RuntimeError("step_moving_frame: CFL substep cap exceeded -- frame velocity diverged")
            if self._decomp is not None:
                self._decomp.exchange(self._u)
                self._decomp.exchange(self._U)          # ghost neighbours' conserved densities
            mu, Te, u = fs.recover_frame(self._U, Te_guess=self._Te)
            uf = self._faces_fn(self._u).reshape(-1, self.Nk)   # reconstruct faces once, reuse
            dU = self._march_U(self._u, mu, Te, u, t, uf)       # KFVS lab flux (rate; dt-free)
            # FULL shell transport (14) grid velocities.  ∂_t(μ,Tₑ,k_D) from the flux-form
            # U-march; (v+u)·∇(μ,Tₑ,k_D) in DIVERGENCE form via the transport's own flux
            # operator (_frame_adv) so the discrete material derivative is same-operator
            # consistent with the streaming (D_mesh k -> 0, free-stream/moment exact).
            qframe = torch.stack([mu, Te, fs.mstar * u[:, 0] / fs.hbar,
                                  fs.mstar * u[:, 1] / fs.hbar], 1)          # (K,4)
            av = self._frame_adv(qframe, mu, Te, u).reshape(self.K, 4, fs.Nr, fs.angular.N_theta)
            dmu, dTe, dkD = fs.dframe_from_dU(dU, mu, Te, u)
            xidot, phidot = fs.shell_velocities(mu, Te, u, dmu, dTe, dkD,
                                                av[:, 0], av[:, 1], av[:, 2], av[:, 3])
            h = min(left, self._cfl_substep(mu, Te, u, xidot, phidot))  # CFL-limited substep
            if not (h > 0.0 and np.isfinite(h)):
                raise RuntimeError("step_moving_frame: non-finite CFL substep -- frame velocity diverged")
            U_new = self._U + h * dU
            Jz = fs.mstar * Te / fs.hbar ** 2                    # 𝒥 (shell-constant per cell)
            Jlo = Jz[:, None, None].expand(self.K, 1, fs.angular.N_theta)
            dG = self._transportG(self._u, uf, Jz, mu, Te, u, xidot, phidot,
                                  Jlo, torch.zeros_like(Jlo), t)  # shape: core 𝒥 / tail 0
            dJv = self._transportG(None, uf, Jz, mu, Te, u, xidot, phidot,
                                   Jlo, Jlo, t)                   # volume GCL: 𝒥 both sides
            f_tr = self._u + h * (dG - self._u * dJv) / Jz[:, None]
            if not self._skip_collision:
                # Collide the MOMENT-FREE deviation about f0 (project removes the four
                # invariants {N,E,px,py}) so C[f0]=0 and no invariant is damped in the
                # sech² metric that the flat projection then fights.  τ_p=∞ untouched.
                d0 = fs.project_moment_free(self._u - fs.rho0, mu, Te)
                f_tr = f_tr + h * fs.rho_dot(d0, t, id(self))
            mu2, Te2, u2 = fs.recover_frame(U_new, Te_guess=Te)
            f_new = fs.pauli_reproject(fs.rho0 + fs.project_moment_free(f_tr - fs.rho0, mu2, Te2),
                                       mu2, Te2)
            if self._owned_mask is not None:
                U_new = torch.where(self._owned_mask, U_new, self._U)
            self._U, self._u, self._Te = U_new, self._dealias(f_new), Te2
            left -= h; t += h

    # ---- moving-frame diagnostics (conservation / consistency) ----
    def U_totals(self) -> torch.Tensor:
        """Domain totals (int n, int Jx, int Jy, int E) = sum_c A_c U_c (owned cells)."""
        sl = slice(self._own_start, self._own_stop)
        tot = (self.geom.area[sl, None] * self._U[sl]).sum(0)
        return self.comm.allreduce(tot) if self._mpi else tot

    def consistency_residual(self) -> torch.Tensor:
        """max_c |moment_of_f(f_c) - U_c| / scale, per channel (owned cells) -- the
        machine-precision check that the shape's moments equal the marched densities."""
        fs = self.material
        sl = slice(self._own_start, self._own_stop)
        mu, Te, u = fs.recover_frame(self._U[sl], Te_guess=self._Te[sl])
        Uf = fs.moments_of_f(self._u[sl], mu, Te, u)
        scale = self._U[sl].abs().mean(0).clamp_min(1e-300)
        res = ((Uf - self._U[sl]).abs() / scale).max(0).values
        return self.comm.allreduce(res) if self._mpi else res

    # ---- qimpy Geometry contract ----
    def rho_dot(self, rho: TensorList, t: float) -> TensorList:
        u = rho[0]
        if self._decomp is not None:
            self._decomp.exchange(u)                          # fill halo ghost rows
        out = self._srhs_fn(u, t)                             # de-alias + spatial RHS (fused)
        if not self._skip_collision:                          # ballistic: collision is exactly 0
            # collision = from_modes(-rates * to_modes(.)); its to_modes already
            # annihilates the ghost, so the raw (un-de-aliased) u is exact here.
            lo, hi = self._own_start, self._own_stop
            out[lo:hi] = out[lo:hi] + self.material.rho_dot(u[lo:hi], t, id(self))
        if self._owned_mask is not None:
            out = out * self._owned_mask
        return TensorList([out])

    @property
    def rho(self) -> TensorList:
        return TensorList([self._u])

    @rho.setter
    def rho(self, rho_new: TensorList) -> None:
        self._u = self._dealias(rho_new[0])

    @property
    def density(self) -> torch.Tensor:
        return self._u

    # ---- contact diagnostics ----
    def contact_currents(self, t: float = 0.0) -> dict[str, float]:
        """Net outward number-current through each contact (positive = out of the
        device). Floating probes read ~0; current sources read their I_set; fixed
        contacts read a response. Sum over all boundaries = -d/dt of mass."""
        if self._decomp is not None:
            self._decomp.exchange(self._u)
        uf = self._faces(self._u).reshape(-1, self.Nk)
        uMb = uf[self.geom.bF]
        uup_b = torch.where(self._maskB, uMb, self._exterior(uMb, t))
        out = {}
        for c in self._contacts:
            I = float((c.cur * uup_b[c.idx]).sum())           # this rank's edges
            out[c.name] = self.comm.allreduce(I) if self._mpi else I
        return out

    def contact_potentials(self) -> dict[str, float]:
        """Self-adjusting level of each feedback contact, from the last evaluation."""
        return {c.name: c.level for c in self._contacts if c.kind != "fixed"}

    def _edge_geometry_static(self):
        """Static per-edge midpoints/normals/lengths, ordered interior-then-boundary
        to match the fv_edge_flux layout (the SAME order _march_U books its interior
        `Ff` then its boundary flux).  Pure numpy on the host; cached."""
        if self._edge_geom_cache is None:
            g, nf = self.geom, self._nf
            fmid = g.face_mid_np                                  # (K, nf, 2)
            eL = g.eL.detach().cpu().numpy(); eLF = g.eLF.detach().cpu().numpy()
            bc = g.bcell.detach().cpu().numpy(); bF = g.bF.detach().cpu().numpy()
            mid_i = fmid[eL, eLF % nf] if len(eL) else np.zeros((0, 2))
            mid_b = fmid[bc, bF % nf] if len(bc) else np.zeros((0, 2))
            mid = np.concatenate([mid_i, mid_b], 0)              # (Nedge, 2)
            nrm = np.concatenate([g.en.detach().cpu().numpy(),
                                  g.bn.detach().cpu().numpy()], 0)    # (Nedge, 2)
            length = np.concatenate([g.elen.detach().cpu().numpy(),
                                     g.blen.detach().cpu().numpy()], 0)  # (Nedge,)
            self._edge_geom_cache = (np.ascontiguousarray(mid, np.float64),
                                     np.ascontiguousarray(nrm, np.float64),
                                     np.ascontiguousarray(length, np.float64))
        return self._edge_geom_cache

    def _stash_moving(self, i_step: int, t: float) -> None:
        """Staggered output for the moving drift-frame scheme.  Records CELL-CENTERED
        scalars [n, T_e, |u|, E] per owned cell AND per-edge normal-flux DENSITIES
        [j.n^, q.n^] at every interior+boundary face midpoint.  The edge fluxes are
        the SAME KFVS lab fluxes the U-march books (captured inside _march_U), not a
        recomputed cell-centered approximation.  MPI: each edge is filled only by the
        rank owning its reference cell (interior -> owner of eL, boundary -> owner of
        bcell), so every edge has exactly one owner and a checkpoint-time SUM across
        ranks assembles the global per-edge array with no double counting."""
        fs, g = self.material, self.geom
        sl = slice(self._own_start, self._own_stop)
        _, Te_o, u_o = fs.recover_frame(self._U[sl], Te_guess=self._Te[sl])
        cell_obs = torch.stack([self._U[sl, 0], Te_o, u_o.norm(dim=1),
                                self._U[sl, 3]], dim=-1)          # (K_own, 4)
        # Per-edge fluxes: reuse _march_U (populates self._Ff_int_dens/_Ff_bnd_dens).
        if self._decomp is not None:
            self._decomp.exchange(self._u)
            self._decomp.exchange(self._U)
        mu, Te, u = fs.recover_frame(self._U, Te_guess=self._Te)
        uf = self._faces_fn(self._u).reshape(-1, self.Nk)
        self._march_U(self._u, mu, Te, u, t, uf)
        edge_flux = torch.cat([self._Ff_int_dens, self._Ff_bnd_dens], 0)  # (Nedge,2)
        if self._mpi:                                            # keep only owned edges
            lo, hi = self._own_start, self._own_stop
            own = torch.cat([(g.eL >= lo) & (g.eL < hi),
                             (g.bcell >= lo) & (g.bcell < hi)], 0)[:, None]
            edge_flux = torch.where(own, edge_flux, torch.zeros_like(edge_flux))
        self._stash_i.append(i_step)
        self._stash_t.append(t)
        self._stash_obs.append(cell_obs.detach().cpu().numpy())
        self._stash_edge.append(edge_flux.detach().cpu().numpy())

    def update_stash(self, i_step: int, t: float) -> None:
        if self._moving:
            self._stash_moving(i_step, t)     # staggered: cell scalars + face fluxes
            return
        # Stash observables for this rank's owned cells (its checkpoint slice).
        u_own = self._u[self._own_start:self._own_stop]
        obs = torch.einsum("oc,kc->ko", self.material.get_observables(t), u_own)
        self._stash_i.append(i_step)
        self._stash_t.append(t)
        self._stash_obs.append(obs.detach().cpu().numpy())
        if self.save_terms and hasattr(self.material, "ee_scattering"):
            # Per-(m,l) collision breakdown, computed WARM in the evolution loop
            # (the same context as the in-run apply) to dodge the cold
            # standalone-apply slow path. a is the modal distribution; lin/quad/
            # cub are the linear, quadratic-in-deltaf and cubic-in-deltaf parts
            # of the e-e operator. All are modal, flattenable to (Nr, dim).
            with torch.no_grad():
                a = self.material.to_modes(u_own)            # (K_own, Nr*dim)
                lin, quad, cub = self.material.ee_scattering.a_dot_breakdown(a)
                self._stash_terms.append(np.stack([
                    a.detach().cpu().numpy(),
                    lin.detach().cpu().numpy(),
                    quad.detach().cpu().numpy(),
                    cub.detach().cpu().numpy(),
                ], axis=0))                                  # (4, K_own, Nr*dim)

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        g = self.geom
        # Moving drift-frame: fv_observables holds CELL-CENTERED scalars only
        # [n, T_e, |u|, E]; the vector currents/heat flux are emitted face-native
        # in fv_edge_flux (below), NOT as cell-centered observables.
        names = (["n", "T_e", "u_mag", "E"] if self._moving
                 else self.material.get_observable_names())
        cp_path.attrs["order"] = 0                            # piecewise-constant FV
        cp_path.attrs["mesh_file"] = self.mesh_file
        saved = [
            cp_path.write("mesh_vertices", torch.from_numpy(g.vertices_np)),
            cp_path.write("mesh_triangles",
                          torch.from_numpy(g.triangles_np.astype(np.int64))),
            cp_path.write("cell_centroid", torch.from_numpy(g.centroid_np)),
            "fv_observables",
        ]
        cp_path.write_str("contact_names", ",".join(self.contacts.keys()))
        cp_path.write_str("observable_names", ",".join(names))
        cp_path["t"] = np.array(self._stash_t)
        cp_path["i_step"] = np.array(self._stash_i)
        checkpoint, path = cp_path
        n_stash = len(self._stash_t)
        # Collective: every rank creates the global dataset, then writes the
        # slice of cells it owns ([own_start, own_stop)). Serially this is the
        # whole array; under MPI it is each rank's contiguous block (needs an
        # mpio-enabled h5py to run multi-rank).
        CheckpointPath(checkpoint, path).create_dataset(
            "fv_observables", (n_stash, self.K, len(names)), np.float64)
        if checkpoint is not None and n_stash:
            checkpoint.write_slice(checkpoint[f"{path}/fv_observables"],
                                   (0, self._own_start, 0),
                                   torch.from_numpy(np.stack(self._stash_obs)))
        if self._moving and len(self._stash_edge):
            # Staggered VECTOR output: per-edge normal-flux densities [j.n^, q.n^] at
            # the face midpoints, plus the static face geometry (midpoints, normals,
            # lengths) so the plotter has faces.  Each edge is owned by exactly one
            # rank (interior -> eL block, boundary -> bcell block); the per-frame
            # arrays are zero off the owner, so an MPI SUM assembles the global array.
            mid, nrm, length = self._edge_geometry_static()
            saved += [cp_path.write("edge_midpoints", torch.from_numpy(mid)),
                      cp_path.write("edge_normals", torch.from_numpy(nrm)),
                      cp_path.write("edge_lengths", torch.from_numpy(length))]
            edge_arr = np.ascontiguousarray(np.stack(self._stash_edge))  # (nstash,Nedge,2)
            if self._mpi:
                out = np.empty_like(edge_arr)
                self.comm.Allreduce(edge_arr, out, op=MPI.SUM)   # one owner per edge
                edge_arr = out
            saved.append(cp_path.write("fv_edge_flux", torch.from_numpy(edge_arr)))
        if self.save_rho:
            # Raw per-cell state (n_cells, n_channels), for exact restart /
            # steady-state warm start. Each rank writes its owned cell block.
            u_own = self._u[self._own_start:self._own_stop].detach().cpu().numpy()
            CheckpointPath(checkpoint, path).create_dataset(
                "rho", (self.K, self.Nk), u_own.dtype)
            if checkpoint is not None:
                checkpoint.write_slice(checkpoint[f"{path}/rho"],
                                       (self._own_start, 0),
                                       torch.from_numpy(u_own))
            saved.append("rho")
        if self.save_terms and len(self._stash_terms):
            # Per-(m,l) collision breakdown stacked over saved frames:
            # (n_stash, 4, K, Nr*dim) with channel order [a, lin, quad, cub].
            # Reshape the last axis to (Nr, dim) offline for the radial/angular
            # (l, m) contributions. dim = 2*M_theta+1 (m = -M..M).
            fs = self.material
            Nr_m = int(fs.Nr); dim_m = int(fs.angular.dim)
            n_modal = Nr_m * dim_m
            cp_path.attrs["terms_Nr"] = Nr_m
            cp_path.attrs["terms_dim"] = dim_m
            cp_path.write_str("terms_channels", "a,lin,quad,cub")
            terms_own = np.stack(self._stash_terms)          # (n_stash,4,K_own,n_modal)
            CheckpointPath(checkpoint, path).create_dataset(
                "fv_terms", (n_stash, 4, self.K, n_modal), terms_own.dtype)
            if checkpoint is not None:
                checkpoint.write_slice(checkpoint[f"{path}/fv_terms"],
                                       (0, 0, self._own_start, 0),
                                       torch.from_numpy(terms_own))
            saved.append("fv_terms")
        self._stash_t, self._stash_i, self._stash_obs = [], [], []
        self._stash_terms = []
        self._stash_edge = []
        return saved
