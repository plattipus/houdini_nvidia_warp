Usage
=====

Drop an **NVIDIA Warp** SOP after any geometry. The shipped defaults are a
ripple deformer, so a new node produces visible output immediately.

Kernel tab
----------

Ordinary Warp kernel source. Every ``@wp.kernel`` defined here is available to
the Warp Code under its own name. The kernel named in **Kernel Name** is also
bound as ``kernel``.

.. code-block:: python

   import warp as wp


   @wp.kernel
   def deform(points: wp.array(dtype=wp.vec3),
              amplitude: float,
              out: wp.array(dtype=wp.vec3)):
       i = wp.tid()
       p = points[i]
       p[1] = p[1] + amplitude
       out[i] = p

The source is written to a temporary module and imported rather than executed
with ``exec()``, because Warp's code generator reads the function's source
through :func:`inspect.getsourcelines`, which fails for code compiled from a
string. Modules are cached on a hash of the source, so an unchanged kernel is
not recompiled between cooks.

A kernel whose name matches one of the names the node provides, such as ``np``
or ``geo``, is rejected rather than allowed to shadow it.

Warp Code tab
-------------

Host-side Python, run once per cook. These names are provided:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Description
   * - ``wp``, ``np``, ``hou``
     - warp, numpy and the Houdini Python module
   * - ``kernel``
     - the compiled kernel named in Kernel Name
   * - ``geo``
     - read-only input geometry, a :class:`hou.Geometry`
   * - ``houdiniGeo``
     - writable geometry, created on first use
   * - ``npoints``
     - point count of the input
   * - ``device``
     - ``"cpu"`` or ``"cuda"``, from the Device parameter
   * - ``out_attrib``
     - the Output Attribute name
   * - ``point_attrib_array(name)``
     - float32 numpy view of a point attribute
   * - ``store``
     - dict that persists between cooks
   * - ``out``
     - assign results here

Returning results
-----------------

Assign ``out``:

.. code-block:: python

   out = {"name": array, ...}   # each key names a point attribute
   out = array                  # shorthand for {out_attrib: array}
   out = {}                     # write nothing

An array shaped ``(npoints,)`` becomes a float attribute and ``(npoints, 3)``
becomes a vector attribute. Any other shape is an error naming the attribute
and the shape received. Values may be numpy arrays, ``wp.array`` objects, or
any sequence of floats.

Leaving ``out`` unassigned is an error, so a typo cannot produce a node that
cooks cleanly and writes nothing.

Writing through the geometry API
--------------------------------

``houdiniGeo`` is a writable :class:`hou.Geometry` supporting the whole HOM
API, including topology changes:

.. code-block:: python

   houdiniGeo.setPointFloatAttribValuesFromString(
       "P", positions, hou.numericData.Float32)

   p = houdiniGeo.createPoint()
   p.setPosition(hou.Vector3(0, 1, 0))

It is created the first time the name is mentioned, and the node then adopts
it as the cook result. A field that never mentions it makes no copy.

``hou.pwd().geometry()`` cannot be used. It asks the node to cook while the
node is already cooking, which Houdini reports as *Infinite recursion in
evaluation*. This applies to any node's own parameters, not only this one. A
Python SOP is the exception because Houdini hands it the working geometry
directly, through a layer that is not part of the public HDK.

Reading other nodes
-------------------

``hou.node("/obj/geo1/some_node").geometry()`` works inside the Warp Code, so
any already-cooked node in the scene can be read. Only this node's own
geometry is unavailable.

Simulations
-----------

The node cooks from its input every time, so state does not accumulate on its
own. Wrap it in a Solver SOP and wire ``Prev_Frame`` to its input to feed each
frame's output back as the next frame's input.

Errors
------

Failures in either code field are reported on the node with the line number
inside the field, and the node returns its input geometry unchanged.
