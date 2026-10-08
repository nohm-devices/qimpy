"""Generate the external triangle mesh for rect-domain-tri.yaml.

qimpy does not mesh; this is the external tooling step. Produces a rect-domain
[5,105]x[5,55] mesh with source/drain contact faces, written in the format
qimpy.transport.geometry.FiniteVolume consumes (see _mesh.load_mesh).

The triangulation is a structured criss-cross grid -- each rectangle split into
four triangles about its centre node -- which is PERFECTLY SYMMETRIC about both
centre lines x=55 and y=30 (no diagonal bias). The source (top) and drain
(bottom) contacts are exact mirror images about y=30, so a +/-dmu drive gives an
exactly antisymmetric steady state: a clean check on the solver.

    python make_rect_mesh.py            # writes rect-domain-tri.npz (h=3.0)
    python make_rect_mesh.py --h 1.5
"""

import argparse

import numpy as np

from qimpy.io import Checkpoint, CheckpointPath
from qimpy.transport.geometry import Mesh

X0, X1, Y0, Y1 = 5.0, 105.0, 5.0, 55.0
SRC, DRN = (10.0, 55.0, 5.0), (10.0, 5.0, 5.0)  # (cx, cy, r) contact circles


def make(h: float, path: str) -> None:
    # Even cell counts put the centre lines x=55, y=30 on node lines (so the
    # mesh is symmetric about them); nx = 2*ny makes the cells square (2:1 domain).
    ny = max(2, 2 * round(0.5 * (Y1 - Y0) / h))
    nx = 2 * ny
    xs = np.linspace(X0, X1, nx + 1)
    ys = np.linspace(Y0, Y1, ny + 1)

    V = [[x, y] for y in ys for x in xs]  # corner nodes (row-major)
    vid = lambda i, j: j * (nx + 1) + i
    T = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            cen = len(V)
            V.append([0.5 * (xs[i] + xs[i + 1]), 0.5 * (ys[j] + ys[j + 1])])
            T += [[a, b, cen], [b, c, cen], [c, d, cen], [d, a, cen]]  # all CCW
    V = np.array(V, float)

    edges = (
        [(vid(i, 0), vid(i + 1, 0)) for i in range(nx)]  # bottom
        + [(vid(i, ny), vid(i + 1, ny)) for i in range(nx)]  # top
        + [(vid(0, j), vid(0, j + 1)) for j in range(ny)]  # left
        + [(vid(nx, j), vid(nx, j + 1)) for j in range(ny)]
    )  # right
    be = []
    for a, b in edges:
        mx, my = 0.5 * (V[a] + V[b])
        if (mx - SRC[0]) ** 2 + (my - SRC[1]) ** 2 <= SRC[2] ** 2:
            boundary_id = 0
        elif (mx - DRN[0]) ** 2 + (my - DRN[1]) ** 2 <= DRN[2] ** 2:
            boundary_id = 1
        else:
            boundary_id = -1
        be.append([a, b, boundary_id])

    mesh = Mesh(file="")
    mesh.vertices = np.array(V)
    mesh.cells = np.hstack((np.array(T), np.zeros((len(T), 1), dtype=int)))
    mesh.edges = np.array(be)
    mesh.region_names = []
    mesh.boundary_names = ["source", "drain"]
    mesh.lattice_vectors = None
    with Checkpoint(path, writable=True) as cp:
        mesh.save(CheckpointPath(cp))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=3.0, help="target cell size")
    ap.add_argument("--out", type=str, default="rect-domain-mesh.h5")
    a = ap.parse_args()
    make(a.h, a.out)
