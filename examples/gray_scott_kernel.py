import warp as wp


@wp.kernel
def rd_step(A: wp.array(dtype=wp.float32), B: wp.array(dtype=wp.float32),
            A_out: wp.array(dtype=wp.float32), B_out: wp.array(dtype=wp.float32),
            offs: wp.array(dtype=wp.int32), nbr: wp.array(dtype=wp.int32),
            DA: float, DB: float, feed: float, kill: float, dt: float):
    """One Gray-Scott step over an arbitrary mesh.

    Neighbours come from a CSR adjacency graph, so the kernel works on any
    topology rather than assuming a regular grid.
    """
    i = wp.tid()
    a = A[i]
    b = B[i]

    s = offs[i]
    e = offs[i + 1]
    n = float(e - s)

    lapA = float(0.0)
    lapB = float(0.0)
    for j in range(s, e):
        p = nbr[j]
        lapA += A[p]
        lapB += B[p]

    if n > 0.0:
        lapA = lapA / n - a
        lapB = lapB / n - b

    r = a * b * b
    A_out[i] = wp.clamp(a + (DA * lapA - r + feed * (1.0 - a)) * dt, 0.0, 1.0)
    B_out[i] = wp.clamp(b + (DB * lapB + r - (kill + feed) * b) * dt, 0.0, 1.0)
