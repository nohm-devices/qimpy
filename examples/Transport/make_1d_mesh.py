"""
Generate a 1D mesh for qimpy.transport. Example:

    python make_1d_mesh.py --nx 64 --out line.h5
"""

import argparse

from qimpy.transport.geometry import Mesh
from qimpy.io import Checkpoint, CheckpointPath

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_file", type=str)
    parser.add_argument("L", nargs="?", type=float, default=1.0)
    parser.add_argument("N", nargs="?", type=int, default=64)
    args = parser.parse_args()
    with Checkpoint(args.mesh_file, writable=True) as cp:
        Mesh.make1D(args.L, args.N).save(CheckpointPath(cp))
