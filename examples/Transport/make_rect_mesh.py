"""
Generate a 2D mesh on a rectangle domain for qimpy.transport of a "channel" gemoetry.
This geometry specifically has contacts at the ends of a pair of opposite edges.
The source-drain transport orientation can be made horizontal or vertical.
Use this as a template to leverage `Mesh.make_rect` for general contact geometries.
Example:

    python make_rect_mesh.py channel-mesh.h5 1.0 1.25 24 30 0.25 horizontal
"""

import argparse

from qimpy.io import Checkpoint, CheckpointPath
from qimpy.transport.geometry import Mesh

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_file", type=str)
    parser.add_argument("Lx", nargs="?", type=float, default=100.0)
    parser.add_argument("Ly", nargs="?", type=float, default=50.0)
    parser.add_argument("Nx", nargs="?", type=int, default=32)
    parser.add_argument("Ny", nargs="?", type=int, default=16)
    parser.add_argument("contact_width", nargs="?", type=float, default=10.0)
    parser.add_argument("orientation", nargs="?", type=str, default="vertical")
    args = parser.parse_args()
    r = 0.5 * args.contact_width  # radius of contact circle
    if args.orientation == "vertical":
        contacts = dict(source=(r, args.Ly, r), drain=(r, 0.0, r))
    else:
        contacts = dict(source=(0.0, r, r), drain=(args.Lx, r, r))
    with Checkpoint(args.mesh_file, writable=True) as cp:
        Mesh.make_rect(args.Lx, args.Ly, args.Nx, args.Ny, contacts).save(
            CheckpointPath(cp)
        )
