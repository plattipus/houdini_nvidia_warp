Performance
===========

Choosing an output path
-----------------------

Two ways to return geometry, with different costs:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Path
     - Cost
     - Use when
   * - ``point_attrib_array`` and ``out``
     - no geometry copy
     - only point attributes change
   * - ``houdiniGeo``
     - one copy in, one adopt out
     - topology changes, or the HOM API is more convenient

Measured on a 300x300 grid, 90,000 points and 89,401 primitives, running one
Gray-Scott step per cook:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Path
     - Cook time
   * - Python SOP doing the same work
     - 155.6 ms
   * - This node through ``houdiniGeo``
     - 145.9 ms
   * - This node through ``geo`` and ``out``
     - 96.0 ms

Materializing ``houdiniGeo`` costs about 0.8 ms at that size, so the choice of
output path rarely dominates a cook.

Caching between cooks
---------------------

The Warp Code namespace is rebuilt on every cook, so assigning into
``globals()`` does not persist and anything cached there is recomputed every
frame. Use ``store``, which is keyed per node and survives between cooks:

.. code-block:: python

   key = (npoints, geo.intrinsicValue("primitivecount"))
   if key not in store:
       store[key] = build_connectivity(geo)
   offsets, neighbours = store[key]

Keying on values derived from the topology means the entry is rebuilt when the
geometry changes.

This matters more than the output path. On a 150x150 grid, rebuilding a
neighbour graph in Python costs about 618 ms per frame while the simulation
itself costs about 24 ms. Caching it in ``store`` reduces the per-frame cost to
about 7 ms.

Bulk attribute access
---------------------

``point_attrib_array(name)`` returns a read-only float32 view of the whole
attribute in one call. ``wp.array()`` accepts a read-only host array, so no
defensive copy is needed.

For writing, ``setPointFloatAttribValuesFromString`` passes a numpy buffer
straight through. It is roughly 18 times faster than
``setPointFloatAttribValues`` with a Python list, which converts every element
individually.

Building connectivity
---------------------

Iterating primitives and vertices in Python is usually the slowest part of a
mesh workload. Connectivity can instead be read in bulk by baking it into
vertex attributes upstream with an Attribute Wrangle in Vertices mode:

.. code-block:: text

   i@vpt = @ptnum;
   i@vpr = @primnum;

and reading both arrays in one call:

.. code-block:: python

   vpt = np.array(geo.vertexIntAttribValues("vpt"), np.int64)
   vpr = np.array(geo.vertexIntAttribValues("vpr"), np.int64)

On a 300x300 grid this reduces building an adjacency graph from about 1541 ms
to about 89 ms.

Kernel compilation
------------------

Warp compiles a kernel the first time it is launched and caches the result on
disk. The first cook after editing a kernel therefore takes noticeably longer
than later cooks. The node caches the imported module on a hash of the source,
bounded to the sixteen most recent, and removes evicted modules from
``sys.modules``.
