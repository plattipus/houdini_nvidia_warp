import numpy as np

# Gray-Scott reaction-diffusion over an arbitrary mesh.
#
# Reads through point_attrib_array() and writes through `out`, so no geometry
# copy is made. Wrap this node in a Solver SOP with Prev_Frame wired to its
# input, otherwise every frame restarts from the same seed.

STEPS = 20
DA, DB = 0.5, 0.25
FEED, KILL, DT = 0.055, 0.062, 1.0
SEED_FRACTION = 0.01

# Neighbour graph, cached in `store` because it depends only on topology.
# globals() is rebuilt every cook and cannot be used for this.
key = ("csr", npoints, geo.intrinsicValue("primitivecount"))

if key in store:
    offsets, nbr = store[key]
else:
    # Vertex attributes i@vpt and i@vpr come from the upstream wrangle, and
    # let the adjacency be built with numpy instead of a Python loop over
    # every primitive.
    vpt = np.array(geo.vertexIntAttribValues("vpt"), np.int64)
    vpr = np.array(geo.vertexIntAttribValues("vpr"), np.int64)

    nxt, nxt_prim = np.roll(vpt, -1), np.roll(vpr, -1)
    same = vpr == nxt_prim
    a, b = vpt[same], nxt[same]

    # Close each polygon: its last vertex connects back to its first.
    starts = np.searchsorted(vpr, np.arange(vpr[-1] + 1))
    ends = np.append(starts[1:], len(vpr)) - 1
    a = np.concatenate([a, vpt[ends]])
    b = np.concatenate([b, vpt[starts]])

    # Symmetric and deduplicated, then converted to CSR.
    src, dst = np.concatenate([a, b]), np.concatenate([b, a])
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    keep = np.ones(len(src), bool)
    keep[1:] = (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])
    src, dst = src[keep], dst[keep]

    offsets = np.zeros(npoints + 1, np.int32)
    np.cumsum(np.bincount(src, minlength=npoints), out=offsets[1:])
    nbr = dst.astype(np.int32)
    store[key] = (offsets, nbr)

# State from the previous frame, or a random seed on the first.
if geo.findPointAttrib("A") is not None:
    A_np = point_attrib_array("A")
    B_np = point_attrib_array("B")
else:
    A_np = np.ones(npoints, np.float32)
    B_np = np.zeros(npoints, np.float32)
    B_np[np.random.default_rng(0).random(npoints) < SEED_FRACTION] = 1.0

with wp.ScopedDevice(device):
    A = wp.array(A_np, dtype=wp.float32)
    B = wp.array(B_np, dtype=wp.float32)
    A_out = wp.zeros(npoints, dtype=wp.float32)
    B_out = wp.zeros(npoints, dtype=wp.float32)
    offs_w = wp.array(offsets, dtype=wp.int32)
    nbr_w = wp.array(nbr, dtype=wp.int32)

    for _ in range(STEPS):
        wp.launch(rd_step, dim=npoints,
                  inputs=[A, B, A_out, B_out, offs_w, nbr_w,
                          DA, DB, FEED, KILL, DT])
        A, A_out = A_out, A
        B, B_out = B_out, B

    out = {"A": A.numpy(), "B": B.numpy()}
