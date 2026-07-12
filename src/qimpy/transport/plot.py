from __future__ import annotations
from typing import Optional
import argparse
import glob
import logging

import matplotlib.pyplot as plt
import numpy as np

from qimpy import rc, log, io
from qimpy.profiler import StopWatch
from qimpy.io import log_config, Checkpoint

# Checkpoint time is in Hartree atomic units; 1 a.u. = ℏ/E_h s.
_PS_PER_AU_TIME = 2.4188843265857e-5

# Staggered output: vectors are now face-native. Scalars (density, temperature)
# are CELL-CENTERED in `fv_observables`; vector fluxes (number-current j, heat
# flux q) are emitted by the solver at triangle EDGE (face) midpoints as their
# normal-flux densities `fv_edge_flux` = [j.n^, q.n^] on the static `edge_midpoints`
# / `edge_normals` faces. `run_finite_volume` reconstructs the cell current from
# those per-face normals by least squares (no cell-centered current is assumed);
# legacy checkpoints without `fv_edge_flux` fall back to the old cell-averaged
# current for compatibility.


def main() -> None:
    log_config()
    rc.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="YAML input file", type=str)
    args = parser.parse_args()
    input_dict = io.dict.key_cleanup(io.yaml.load(args.input_file))
    run(**input_dict)

    rc.report_end()
    StopWatch.print_stats()


def run(
    *,
    checkpoints: str,
    output: str,
    density: Optional[dict] = None,
    streamlines: Optional[dict] = None,
    dpi: int = 200,
    **ignored,
) -> None:
    """Render finite-volume (FiniteVolume) transport checkpoints.

    ``**ignored`` absorbs legacy keys from older input files for compatibility.
    """
    if density is None:
        density = {}
    # Distribute frames over MPI:
    file_list = rc.comm.bcast(sorted(glob.glob(checkpoints)))
    mine = slice(rc.i_proc, None, rc.n_procs)
    with Checkpoint(file_list[0]) as cp:
        geom_type = cp["/geometry"].attrs.get("variant_name", b"")
        geom_type = (geom_type.decode() if isinstance(geom_type, bytes)
                     else str(geom_type))
    if geom_type != "spatial_transport":
        raise ValueError(
            "qimpy.transport.plot renders the finite-volume 'spatial_transport'"
            f" geometry; checkpoint has variant_name={geom_type!r}."
        )
    run_finite_volume(file_list, mine, output, density, streamlines, dpi)


def fv_edge_geometry(verts, tris):
    """Staggered-grid face geometry for a triangular finite-volume mesh.

    Each interior edge is shared by two triangles and each boundary edge by
    one. Returns, for every unique edge, its midpoint and the indices of the
    (one or two) adjacent cells -- the ingredients needed to place the current
    on the faces rather than the cells. Boundary edges (a single adjacent cell)
    are also returned as line segments ``bsegs`` for drawing the device outline.
    Pure numpy, computed once per mesh."""
    from collections import defaultdict
    edge_cells = defaultdict(list)
    for k in range(tris.shape[0]):
        a, b, c = int(tris[k, 0]), int(tris[k, 1]), int(tris[k, 2])
        for u, v in ((a, b), (b, c), (c, a)):
            edge_cells[(u, v) if u < v else (v, u)].append(k)
    keys = np.array(list(edge_cells.keys()))                 # (Ne, 2) vert ids
    cell0 = np.array([cs[0] for cs in edge_cells.values()])
    cell1 = np.array([cs[1] if len(cs) == 2 else cs[0]       # boundary: dup
                      for cs in edge_cells.values()])
    emid = 0.5 * (verts[keys[:, 0]] + verts[keys[:, 1]])     # (Ne, 2)
    boundary = np.array([len(cs) == 1 for cs in edge_cells.values()])
    bkeys = keys[boundary]                                   # (Nb, 2) vert ids
    bsegs = np.stack([verts[keys[:, 0]],                     # (Nb, 2, 2): the
                      verts[keys[:, 1]]], axis=1)[boundary]  # device outline
    bcell0 = cell0[boundary]                                 # adj cell per bedge
    return dict(emid=emid, cell0=cell0, cell1=cell1, bsegs=bsegs, bkeys=bkeys,
                bcell0=bcell0)


def fv_face_to_cell(verts, tris, edge_mid, edge_nrm):
    """Static least-squares operator: cell-centered vector from per-face normal data.

    The solver emits vector fluxes as scalar normal-flux densities ``F.n^`` at the
    face (edge) midpoints (staggered grid). To trace streamlines we need a cell
    vector, so per triangle we solve the small least-squares system that best
    matches its three face normals,
        j_c = (sum_f n^_f (x) n^_f)^{-1} sum_f (F.n^)_f n^_f,
    which is exact for a locally uniform field (3 faces, 2 unknowns, rank 2). Only
    the geometry is precomputed here (matched once via a midpoint KD-tree to the
    solver's own edge list, so the stored normal's SIGN is honoured); the per-frame
    values plug into :func:`fv_reconstruct_cell`. Returns ``(face_edge (K,3) int,
    face_nrm (K,3,2), Minv (K,2,2))``. Pure numpy, computed once per mesh."""
    from scipy.spatial import cKDTree
    K = tris.shape[0]
    tree = cKDTree(np.asarray(edge_mid, float))
    loc = np.array([[0, 1], [1, 2], [2, 0]])                 # triangle's 3 local faces
    face_edge = np.zeros((K, 3), int)
    face_nrm = np.zeros((K, 3, 2))
    Minv = np.zeros((K, 2, 2))
    for k in range(K):
        M = np.zeros((2, 2))
        for f in range(3):
            mid = 0.5 * (verts[tris[k, loc[f, 0]]] + verts[tris[k, loc[f, 1]]])
            j = int(tree.query(mid)[1])                      # matching solver edge
            nhat = np.asarray(edge_nrm[j], float)
            face_edge[k, f] = j
            face_nrm[k, f] = nhat
            M += np.outer(nhat, nhat)
        Minv[k] = np.linalg.inv(M + 1e-30 * np.eye(2))       # rank-2, tiny ridge
    return face_edge, face_nrm, Minv


def fv_reconstruct_cell(Fn, face_edge, face_nrm, Minv):
    """Cell-centered vector (K,2) from per-face normal-flux densities ``Fn`` (Nedge,)
    via the precomputed least-squares operator (see :func:`fv_face_to_cell`)."""
    b = np.einsum("kf,kfd->kd", Fn[face_edge], face_nrm)     # sum_f (F.n^)_f n^_f
    return np.einsum("kde,ke->kd", Minv, b)                  # (K, 2)


def fv_contact_mask(bkeys, boundary_edges, boundary_markers,
                    contact_names=None, bmid=None, mesh_vertices=None):
    """Label the boundary edges, flagging contacts vs wall (any geometry).

    The checkpoint records ``contact_names`` (e.g. "source,drain") but not a
    per-edge label, so the per-edge marker comes from the mesh's own
    ``boundary_markers``. When ``contact_names`` is given it is the authoritative
    set of contact markers and *only* those are flagged -- any other marker
    (including non-"wall" sentinels like "insulator"/"gate") is treated as a
    wall. Without it, the legacy rule (anything != "wall" is a contact) applies.

    Edges are matched to the checkpoint's ``bkeys`` by unordered vertex-id pair.
    If the mesh npz and the checkpoint disagree on vertex ordering (so *no* id
    pair matches) and ``bmid``/``mesh_vertices`` are supplied, fall back to a
    coordinate match: nearest mesh edge-midpoint to each ``bmid`` within a small
    tolerance (cKDTree). Returns ``(mask, labels)`` aligned with ``bkeys`` /
    ``bsegs``: ``mask`` True on contacts, ``labels`` the per-edge marker name
    ("wall" where unmatched or non-contact)."""
    names = ({str(c).strip().lower() for c in contact_names}
             if contact_names is not None else None)

    def is_contact(marker):
        s = str(marker).strip().lower()
        return (s in names) if names is not None else (str(marker) != "wall")

    markers = [str(m) for m in boundary_markers]
    # Primary: match by unordered vertex-id pair.
    label = {frozenset((int(a), int(b))): m
             for (a, b), m in zip(boundary_edges, markers)}
    labels = np.array([label.get(frozenset((int(u), int(v))), "wall")
                       for u, v in bkeys], dtype=object)
    mask = np.array([is_contact(m) for m in labels], dtype=bool)
    # Fallback: vertex ids disagree (reindexed mesh) -> match by coordinate.
    if not mask.any() and bmid is not None and mesh_vertices is not None:
        from scipy.spatial import cKDTree
        mv = np.asarray(mesh_vertices, dtype=float)
        be = np.asarray(boundary_edges)
        mesh_mid = 0.5 * (mv[be[:, 0]] + mv[be[:, 1]])
        tol = 1e-6 * float(mv.max() - mv.min() + 1.0)
        dist, idx = cKDTree(mesh_mid).query(np.asarray(bmid, dtype=float))
        hit = dist <= tol
        labels = np.array([markers[idx[i]] if hit[i] else "wall"
                           for i in range(len(bkeys))], dtype=object)
        mask = np.array([is_contact(m) for m in labels], dtype=bool)
    return mask, labels


def fv_contact_annotations(bkeys, bsegs, bcell0, cell_cent, contact,
                           contact_labels, span, pad=0.012):
    """One text anchor ``(name, x, y, angle)`` per *connected* contact pad.

    Robust to arbitrary geometries: contacts sharing a marker name but
    geometrically disjoint (separate pads) are split into connected components
    by shared vertices, so each pad is labelled on its own. "Outward" for each
    component is the mean of its per-edge geometric normals, each flipped to
    point away from that edge's single adjacent cell centroid -- correct on
    concave or interior boundaries where a global mesh-centroid direction would
    point the wrong way. The glyph is rotated to the pad's principal (PCA)
    direction, turned to face outward, then kept right-side-up. The anchor sits
    ``pad*span`` outside the pad-edge centroid."""
    idx = np.nonzero(contact)[0]
    if len(idx) == 0:
        return []
    # Union-find over contact edges joined by shared vertices -> pads.
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    vert_edge: dict = {}
    for i in idx:
        find(int(i))                                 # register
        for w in (int(bkeys[i, 0]), int(bkeys[i, 1])):
            if w in vert_edge:
                union(int(i), vert_edge[w])
            vert_edge[w] = int(i)
    comps: dict = {}
    for i in idx:
        comps.setdefault(find(int(i)), []).append(int(i))

    out = []
    for members in comps.values():
        members = np.array(members)
        segs = bsegs[members]                        # (m, 2, 2)
        p0, p1 = segs[:, 0, :], segs[:, 1, :]
        emid = 0.5 * (p0 + p1)
        # Per-edge geometric outward normal, flipped away from cell centroid.
        nrm = np.stack([p1[:, 1] - p0[:, 1], p0[:, 0] - p1[:, 0]], axis=1)
        to_cell = cell_cent[bcell0[members]] - emid
        flip = np.sum(nrm * to_cell, axis=1) > 0.0   # normal points inward
        nrm[flip] = -nrm[flip]
        d = nrm.sum(axis=0)
        nd = float(np.hypot(d[0], d[1]))
        d = d / nd if nd else np.array([0.0, 1.0])   # outward unit vector
        # Principal (PCA) direction of the pad's endpoints -> text rotation.
        pts = segs.reshape(-1, 2)
        pp = pts - pts.mean(axis=0)
        t_hat = np.linalg.eigh(pp.T @ pp)[1][:, -1]
        ang = np.degrees(np.arctan2(t_hat[1], t_hat[0]))
        up = np.array([-t_hat[1], t_hat[0]])         # glyph-up when text || t_hat
        if up @ d < 0.0:                             # turn the text to face out
            up, ang = -up, ang + 180.0
        if up[1] < -1e-9:                            # ... but keep it upright
            ang += 180.0
        labs = np.array([str(s) for s in contact_labels[members]])
        uniq, cnt = np.unique(labs, return_counts=True)
        name = str(uniq[cnt.argmax()])               # majority marker in pad
        c = emid.mean(axis=0)
        x, y = c + pad * span * d
        out.append((name, float(x), float(y), float(ang)))
    return out


def _read_contact_names(g):
    """Best-effort read of the checkpoint's ``contact_names`` (or None).

    Stored either as one comma-joined string or an array of names; returns a
    flat list of individual names, or None if absent/unreadable so the caller
    falls back to the legacy "anything != wall is a contact" rule."""
    try:
        arr = np.array(g["contact_names"])
    except Exception:
        return None
    parts = [arr.item()] if arr.ndim == 0 or arr.size == 1 else list(arr.ravel())
    names = []
    for p in parts:
        s = p.decode() if isinstance(p, (bytes, bytearray)) else str(p)
        names += [q for q in s.replace(",", " ").split() if q]
    return names or None


def _read_str_list(g, key):
    """Read a comma-joined string dataset (e.g. ``observable_names``) into an
    ORDER-PRESERVING list of labels, or None if absent. Used to map a requested
    scalar field to its column in ``fv_observables``."""
    try:
        arr = np.array(g[key])
    except Exception:
        return None
    s = arr.item()
    s = s.decode() if isinstance(s, (bytes, bytearray)) else str(s)
    labels = [q.strip() for q in s.split(",") if q.strip()]
    return labels or None


def run_finite_volume(file_list, mine, output, density, streamlines, dpi) -> None:
    """Frame-parallel, mesh-native rendering of FiniteVolume (finite-volume) output.

    Scalars are cell-centred: the chosen field (density by default) is drawn as a
    flat-shaded ``tripcolor`` (piecewise constant, the honest FV picture) over the
    actual mesh. Vectors are FACE-NATIVE: the solver emits the current/heat-flux
    normal densities at the triangle edge midpoints (``fv_edge_flux`` on
    ``edge_midpoints``/``edge_normals``); streamlines are traced from a cell current
    reconstructed by least squares from those per-face normals (``fv_face_to_cell``)
    -- no cell-centred current is assumed. Older checkpoints without ``fv_edge_flux``
    fall back to the previous cell-averaged current. Each rank renders its strided
    subset of frames, so post-processing scales like the solve."""
    import os
    import matplotlib.tri as mtri
    from matplotlib.collections import LineCollection
    from scipy.interpolate import griddata
    cmap = density.get("cmap", "bwr")
    with Checkpoint(file_list[0]) as cp:
        g = cp["/geometry"]
        verts = np.array(g["mesh_vertices"])         # (Nv, 2)
        tris = np.array(g["mesh_triangles"])         # (K, 3)
        mesh_file = g.attrs.get("mesh_file", b"")
        contact_names = _read_contact_names(g)       # authoritative set or None
        obs_names = _read_str_list(g, "observable_names")   # column labels or None
        # Face-native vectors: present iff the solver emitted the staggered output.
        has_faces = "fv_edge_flux" in g
        edge_mid = np.array(g["edge_midpoints"]) if has_faces else None   # (Nedge, 2)
        edge_nrm = np.array(g["edge_normals"]) if has_faces else None     # (Nedge, 2)
    mesh_file = (mesh_file.decode() if isinstance(mesh_file, bytes)
                 else str(mesh_file))
    triang = mtri.Triangulation(verts[:, 0], verts[:, 1], tris)
    edges = fv_edge_geometry(verts, tris)            # face (edge) geometry, once
    bmid = edges["bsegs"].mean(axis=1)               # (Nb, 2) edge midpoints
    cell_cent = verts[tris].mean(axis=1)             # (K, 2) triangle centroids
    span = max(verts[:, 0].max() - verts[:, 0].min(),
               verts[:, 1].max() - verts[:, 1].min())
    # Cell-centred scalar to shade (density by default); mapped to its stored column.
    field = density.get("field", "n")
    fcol = obs_names.index(field) if (obs_names and field in obs_names) else 0
    # Precompute the static face->cell least-squares operator for the vectors
    # (only needed when tracing reconstructed streamlines).
    if has_faces and streamlines is not None:
        face_edge, face_nrm, Minv = fv_face_to_cell(verts, tris, edge_mid, edge_nrm)
    # Contact edges (gold) vs walls (black), from the mesh's own markers.
    contact = np.zeros(len(edges["bkeys"]), dtype=bool)
    contact_labels = np.full(len(edges["bkeys"]), "wall", dtype=object)
    base = os.path.basename(mesh_file) if mesh_file else ""
    here = os.path.dirname(os.path.abspath(file_list[0]))
    for cand in ([mesh_file, os.path.join(here, base), base] if mesh_file else []):
        if cand and os.path.exists(cand):
            mz = np.load(cand, allow_pickle=True)    # trusted: our own mesh
            if "boundary_edges" in mz and "boundary_markers" in mz:
                contact, contact_labels = fv_contact_mask(
                    edges["bkeys"], mz["boundary_edges"], mz["boundary_markers"],
                    contact_names=contact_names, bmid=bmid,
                    mesh_vertices=mz["vertices"] if "vertices" in mz else None)
            break
    # One text anchor per connected contact pad, just outside its own edge.
    contact_text = fv_contact_annotations(
        edges["bkeys"], edges["bsegs"], edges["bcell0"], cell_cent,
        contact, contact_labels, span, density.get("contact_pad", 0.012))
    if streamlines is not None:
        xs = np.linspace(verts[:, 0].min(), verts[:, 0].max(), 220)
        ys = np.linspace(verts[:, 1].min(), verts[:, 1].max(), 220)
        Xs, Ys = np.meshgrid(xs, ys)
        inside = triang.get_trifinder()(Xs, Ys) >= 0  # grid points within mesh
    orig_level = log.getEffectiveLevel(); log.setLevel(logging.INFO)
    for checkpoint_file in file_list:
        with Checkpoint(checkpoint_file) as cp:
            g = cp["/geometry"]
            i_step_list = np.array(g["i_step"])[mine]
            t_list = np.array(g["t"])[mine]
            obs = np.array(g["fv_observables"][mine])   # (nframe, K, n_scalar)
            edge_flux = (np.array(g["fv_edge_flux"][mine])   # (nframe, Nedge, 2)
                         if has_faces else None)
        for fr, (i_step, t) in enumerate(zip(i_step_list, t_list)):
            n_val = obs[fr, :, fcol]                     # (K,) per-cell scalar
            vmax = float(np.nanmax(np.abs(n_val)))
            if not np.isfinite(vmax) or vmax == 0.0:
                vmax = 1.0
            fig, ax = plt.subplots(figsize=(6, 6))
            tpc = ax.tripcolor(triang, facecolors=n_val / vmax, cmap=cmap,
                               vmin=-1, vmax=1)         # flat shading = FV cell average
            ax.set_aspect("equal")
            ax.set_title(f"$t$ = {t * _PS_PER_AU_TIME:.4g} ps")
            ax.axis("off")
            lw = density.get("outline_lw", 2.0)
            ax.add_collection(LineCollection(                # walls (black)
                edges["bsegs"][~contact], colors="k", linewidths=lw, zorder=5))
            if contact.any():                                # source/drain (gold)
                gold = density.get("contact_color", "gold")
                ax.add_collection(LineCollection(
                    edges["bsegs"][contact], colors=gold, capstyle="round",
                    linewidths=density.get("contact_lw", 7.5), zorder=6))
                for name, tx, ty, ang in contact_text:       # label each contact
                    ax.text(tx, ty, name, ha="center", va="center", zorder=7,
                            rotation=ang, rotation_mode="anchor",
                            fontsize=density.get("contact_fontsize", 9),
                            fontweight="bold", color="black",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                      ec=gold, alpha=0.85, lw=1.0))
            cb = fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)
            label = "Density" if fcol == 0 else field
            cb.set_label(rf"{label} ($\times$max = {vmax:.2e})")
            if streamlines is not None and (has_faces or obs.shape[-1] >= 3):
                jx = jy = None
                if has_faces:
                    # FACE-NATIVE current: stored as its normal-flux density j.n^ at
                    # the edge midpoints. mode="quiver" draws those face normals
                    # directly; otherwise reconstruct the cell current by least
                    # squares from the per-face normals (fv_face_to_cell) and trace
                    # it -- no cell-centred current is stored or assumed.
                    Fn = edge_flux[fr, :, 0]                 # number-current j.n^
                    if streamlines.get("mode") == "quiver":
                        ax.quiver(edge_mid[:, 0], edge_mid[:, 1],
                                  Fn * edge_nrm[:, 0], Fn * edge_nrm[:, 1],
                                  color="k", zorder=4, pivot="mid", angles="xy",
                                  width=streamlines.get("width", 0.003),
                                  scale=streamlines.get("quiver_scale", None))
                    else:
                        Jc = fv_reconstruct_cell(Fn, face_edge, face_nrm, Minv)
                        jx, jy = Jc[:, 0], Jc[:, 1]
                else:
                    # Legacy checkpoint: current stored cell-averaged (centroids).
                    jx, jy = obs[fr, :, 1], obs[fr, :, 2]
                if jx is not None:
                    U = griddata(cell_cent, jx, (Xs, Ys), method="linear")
                    V = griddata(cell_cent, jy, (Xs, Ys), method="linear")
                    U = np.where(inside, np.nan_to_num(U), np.nan)
                    V = np.where(inside, np.nan_to_num(V), np.nan)
                    ax.streamplot(xs, ys, U, V,
                                  density=streamlines.get("density", 1.5),
                                  linewidth=streamlines.get("linewidth", 0.9),
                                  arrowsize=streamlines.get("arrowsize", 0.9), color="k")
            plot_file = output.format(i_step)
            fig.savefig(plot_file, bbox_inches="tight", dpi=dpi)
            plt.close(fig)
            log.info(f"Saved {plot_file}")
    log.setLevel(orig_level)
    rc.comm.Barrier()


def split_names(input: str) -> list[str]:
    return input.split(",") if input else []


if __name__ == "__main__":
    main()
