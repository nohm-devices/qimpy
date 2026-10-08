"""Generate a 1D line-domain mesh for qimpy.transport (FiniteVolume).

Domain [0, Lx] split into nx interval cells on a line (y = 0): a 1D wire with a
source contact at the left end (x=0) and a drain at the right end (x=Lx).  Cells
are intervals; FiniteVolume's 1D path treats each with two endpoint faces (length is
the cell measure, +/- x the face normals).  The material is unchanged -- the
Fermi surface stays 2D, and only v_x = v.n streams along the line.

    python make_1d_mesh.py --nx 64 --out line.npz
"""

import argparse

from qimpy.transport.geometry import Mesh
from qimpy.io import Checkpoint, CheckpointPath

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--Lx", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="line.h5")
    a = ap.parse_args()
    with Checkpoint(a.out, writable=True) as cp:
        Mesh.make1D(a.Lx, a.nx).save(CheckpointPath(cp))
