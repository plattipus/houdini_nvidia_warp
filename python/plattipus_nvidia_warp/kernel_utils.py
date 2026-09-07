"""Helpers for the plattipus::nvidia_warp SOP.

This module holds the Python that the compiled SOP calls into: kernel
calls into to compile a Warp kernel, run the host-side "Warp Code" field and
hand the results back to C++. It is deliberately not a plugin or solver
registry; the node owns the workflow, this file just holds the Warp-specific
glue that is far more readable in Python than in the CPython C API.

Two code fields
---------------

The node has two large Python fields, like Houdini's Python / OpenCL SOPs:

``Kernel``
    The ``@wp.kernel`` device code. Compiled by :func:`import_kernel_module`,
    which
    writes it to a file and imports it. Warp's code generator calls
    ``inspect.getsourcelines()`` and rejects kernels produced by ``exec()``.

``Warp Code``
    The host-side Python that reads geometry, builds ``wp.array``s, calls
    ``wp.launch()`` and produces the results. Run by :func:`exec_warp_code`
    in a namespace built by :func:`run_warp_code`. Ordinary ``exec()`` is
    fine here: nothing inspects this code's source.

The Warp Code contract
----------------------

Names made available to the Warp Code field:

============================  ==============================================
``wp``                        the ``warp`` module, already initialised
``np``                        ``numpy``
``hou``                       the ``hou`` module
``kernel``                    the kernel named in the Kernel Name parm
``geo``                       read-only ``hou.Geometry`` of input 0
``houdiniGeo``                writable geometry, created on first use
``npoints``                   point count, as an ``int``
``device``                    ``"cpu"`` or ``"cuda"``, from the Device parm
``out_attrib``                the Output Attribute parm, as a ``str``
``point_attrib_array(name)``  bulk numpy read of a float point attribute
``store``                     dict that persists between cooks
``out``                       assign results here
============================  ==============================================

Every ``@wp.kernel`` the Kernel field defines is also bound under its own
name, so ``wp.launch(my_kernel, ...)`` works directly.

The field returns results by assigning ``out``:

* ``out = {"name": array, ...}``: each key names a point attribute.
* ``out = array``: shorthand for ``{out_attrib: array}``.
* ``out = {}``: write nothing. This is not an error.

Leaving ``out`` unassigned (or ``None``) *is* an error, so that a typo in the
field does not silently produce a node that cooks clean and does nothing.

Each value may be a numpy array, a ``wp.array`` (converted with ``.numpy()``)
or anything convertible to a float32 array. The shape decides the attribute:

* ``(npoints,)`` or ``(npoints, 1)`` -> float point attribute
* ``(npoints, 3)`` -> vector point attribute

Anything else is a ValueError naming the attribute and its shape.

Why the write side goes back through C++
----------------------------------------

``HOMF``, the ``GU_Detail`` to ``HOM_Geometry`` bridge, is not shipped in
the public HDK toolkit, so there is no public way to wrap the cooking ``gdp``
in a writable ``hou.Geometry``. Reading is fine (the *input* node's geometry
is reachable and read-only), but every write has to be marshalled back to
``SOP_NvidiaWarp::cookMySop``, which does it with ``GA_RWHandleF`` /
``GA_RWHandleV3``. That is what :func:`marshal_results` exists for.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
import traceback
from collections import OrderedDict

__all__ = [
    "validate_kernel",
    "import_kernel_module",
    "kernels_in_module",
    "input_geometry",
    "point_attrib_array",
    "exec_warp_code",
    "marshal_results",
    "node_store",
    "run_warp_code",
]

# The pseudo-filename the Warp Code field is compiled under. Used to pick the
# user's own frame out of a traceback so runtime errors can report a line
# number that means something in the parameter editor.
WARP_CODE_FILENAME = "<nvidia_warp Warp Code>"

KERNEL_FILENAME = "<nvidia_warp Kernel>"

# Point attribute tuple sizes cookMySop knows how to write. 1 -> GA_RWHandleF,
# 3 -> GA_RWHandleV3. Keep in sync with SOP_NvidiaWarp::writeResultItem().
SUPPORTED_TUPLE_SIZES = (1, 3)

# Names the node itself puts in the Warp Code namespace. A kernel named after
# one of these would shadow it, so a kernel that collides with one is
# refused with an explicit error.
RESERVED_NAMES = frozenset((
    "wp", "np", "hou", "geo", "kernel", "npoints", "device", "out_attrib",
    "point_attrib_array", "out", "houdiniGeo", "store",
))

# Maps a hash of the kernel source to its imported module, so an
# unchanged kernel is not re-imported on every cook. Bounded, because an
# artist editing a kernel in the parameter editor produces a new entry on
# every keystroke-triggered cook; unbounded, this would pin an ever-growing
# set of modules in sys.modules for the life of the session.
_MODULE_CACHE = OrderedDict()
_MODULE_CACHE_SIZE = 16


def _cache_module(digest, module_name, module):
    """Insert into the module cache, evicting the least recently used."""
    _MODULE_CACHE[digest] = (module_name, module)
    _MODULE_CACHE.move_to_end(digest)
    while len(_MODULE_CACHE) > _MODULE_CACHE_SIZE:
        _, (evicted_name, _unused) = _MODULE_CACHE.popitem(last=False)
        sys.modules.pop(evicted_name, None)


# ---------------------------------------------------------------------------
# Kernel field
# ---------------------------------------------------------------------------

def validate_kernel(source, kernel_name):
    """Compile-check *source* and confirm it defines *kernel_name*.

    Returns the compiled code object. Raises ValueError with a message
    suitable for display on the node if the source is unusable.
    """
    if not source or not source.strip():
        raise ValueError("kernel source is empty")

    try:
        code = compile(source, KERNEL_FILENAME, "exec")
    except SyntaxError as exc:
        raise ValueError(
            "syntax error in kernel source, line %s: %s"
            % (exc.lineno, exc.msg)
        ) from exc

    if kernel_name not in code.co_names:
        raise ValueError(
            "kernel source does not define %r" % (kernel_name,)
        )

    return code


def import_kernel_module(source, kernel_name):
    """Import the Kernel field as a module and return it.

    Warp rejects a kernel produced by ``exec()``: its code generator calls
    ``inspect.getsourcelines()`` on the decorated function, which fails for
    code compiled from a string. The source is therefore written to a file and
    imported, as Warp documents.

    Modules are cached on a hash of the source, so re-cooking an unchanged
    kernel does not rewrite the file or re-run Warp's code generation.
    """
    validate_kernel(source, kernel_name)

    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()

    cached = _MODULE_CACHE.get(digest)
    if cached is not None:
        _MODULE_CACHE.move_to_end(digest)
        module = cached[1]
    else:
        module_name = "plattipus_nvidia_warp_kernel_%s" % digest
        directory = os.path.join(
            tempfile.gettempdir(), "plattipus_nvidia_warp_kernels"
        )
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, module_name + ".py")

        if not os.path.exists(path):
            # Write via a unique temp file and rename, so two Houdini
            # sessions cooking the same kernel at once cannot leave a
            # partially-written file for the other to import.
            tmp = "%s.%d.tmp" % (path, os.getpid())
            with open(tmp, "w") as handle:
                handle.write(source)
            os.replace(tmp, path)

        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)

        # inspect.getsourcelines() resolves the module through sys.modules,
        # so it has to be registered before the Warp decorators run.
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise ValueError(
                "kernel source failed to execute: %s: %s"
                % (type(exc).__name__, exc)
            ) from exc

        _cache_module(digest, module_name, module)

    return module


# ---------------------------------------------------------------------------
# Geometry reads
# ---------------------------------------------------------------------------

def _float32_type():
    """Return ``hou.numericData.Float32``.

    Isolated in its own function so the helpers below can be unit-tested on a
    machine with no Houdini by monkeypatching this one call.
    """
    import hou

    return hou.numericData.Float32


def input_geometry(node_path):
    """Return the read-only ``hou.Geometry`` feeding the node at *node_path*.

    The node path is passed in from ``cookMySop`` (``OP_Node::getFullPath()``)
    rather than being discovered with ``hou.pwd()``: an explicit path does not
    depend on HOM's "current node" bookkeeping and is trivial to stub in
    tests.

    NEVER call ``.geometry()`` on the Warp SOP itself. It is the node that is
    currently cooking, so HOM raises "Infinite recursion in evaluation" and
    returns nothing useful. Only the *input* node's geometry is reachable, and
    only for reading. See the module docstring.
    """
    import hou

    if not node_path:
        raise ValueError("no node path was supplied to input_geometry()")

    node = hou.node(node_path)
    if node is None:
        raise ValueError("could not resolve node %r" % (node_path,))

    inputs = node.inputs()
    if not inputs or inputs[0] is None:
        raise ValueError("the NVIDIA Warp SOP has no input connected")

    return inputs[0].geometry()


def point_attrib_array(geo, name):
    """Bulk-read the float point attribute *name* from *geo* as numpy.

    This is the fast path: Houdini hands back the whole attribute as one
    buffer and numpy wraps it, instead of the O(n) per-point round trip
    through Python objects.

    The result is a ``(npoints, size)`` array for tuple sizes above 1 and a
    flat ``(npoints,)`` array for scalars, always float32.

    The array is a read-only view of the buffer Houdini returned, so one copy
    total, not two. ``wp.array()`` accepts read-only host arrays (verified
    against Warp 1.17 for both ``wp.vec3`` and scalar dtypes), so no defensive
    copy is made. Call ``.copy()`` yourself if you need to mutate it in place.
    """
    import numpy

    attrib = geo.findPointAttrib(name)
    if attrib is None:
        raise ValueError(
            "input geometry has no point attribute %r" % (name,)
        )

    size = attrib.size()

    try:
        raw = geo.pointFloatAttribValuesAsString(
            name, float_type=_float32_type()
        )
    except Exception as exc:
        # hou.OperationFailed for non-float attributes. Catching it by name
        # would mean importing hou purely to name an exception class, which
        # would make this helper untestable without Houdini.
        raise ValueError(
            "point attribute %r could not be read as float32: %s"
            % (name, exc)
        ) from exc

    array = numpy.frombuffer(raw, dtype=numpy.float32)
    if size > 1:
        array = array.reshape(-1, size)
    return array


# ---------------------------------------------------------------------------
# Warp Code field
# ---------------------------------------------------------------------------

def exec_warp_code(source, namespace):
    """Compile and run the host-side Warp Code field in *namespace*.

    Returns *namespace*, which the caller then reads ``out`` back out of.
    Every failure is converted to a ValueError carrying a line number that
    refers to the parameter field, because that is the only line numbering an
    artist looking at the node can act on.
    """
    if not source or not source.strip():
        raise ValueError("Warp Code is empty")

    try:
        code = compile(source, WARP_CODE_FILENAME, "exec")
    except SyntaxError as exc:
        raise ValueError(
            "syntax error in Warp Code, line %s: %s" % (exc.lineno, exc.msg)
        ) from exc

    try:
        exec(code, namespace)
    except Exception as exc:
        raise ValueError(
            "Warp Code failed%s: %s: %s"
            % (_warp_code_location(exc), type(exc).__name__, exc)
        ) from exc

    return namespace


def _warp_code_location(exc):
    """Return ``" at line N"`` for the deepest frame inside the Warp Code.

    A traceback from the field is mostly Warp and numpy frames; only the
    frames compiled under WARP_CODE_FILENAME have line numbers the artist can
    map onto what they typed. Returns an empty string if the exception never
    passed through the field (which happens when Warp raises during import).
    """
    lineno = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == WARP_CODE_FILENAME:
            lineno = frame.lineno
    return "" if lineno is None else " at line %d" % lineno


def marshal_results(out, npoints, out_attrib):
    """Turn the Warp Code field's ``out`` into the wire format C++ reads.

    Returns a list of ``(name_utf8_bytes, tuple_size, float32_bytes)``
    triples, in the order the field produced them.

    Names come back as *bytes* on purpose. It means ``cookMySop`` reads every
    string through ``PY_PyBytes_AsStringAndSize``, which is an explicitly
    declared wrapper with an unambiguous signature, instead of leaning on the
    ``PyString_*`` Python-2 compatibility shim to do the right thing on a
    Python 3 ``str``.
    """
    import numpy

    if out is None:
        raise ValueError(
            "Warp Code did not assign `out`. Assign a dict of "
            "{attribute_name: array}, or a single array to write to the "
            "Output Attribute, or {} to write nothing."
        )

    if not isinstance(out, dict):
        # Bare array shorthand: write it to the Output Attribute parm.
        out = {out_attrib: out}

    results = []
    for name, value in out.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "out keys must be non-empty attribute names, got %r"
                % (name,)
            )

        # A wp.array carries its own .numpy(); numpy arrays do not.
        numpy_fn = getattr(value, "numpy", None)
        if callable(numpy_fn):
            value = numpy_fn()

        try:
            array = numpy.ascontiguousarray(value, dtype=numpy.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "out[%r] is not convertible to a float array: %s"
                % (name, exc)
            ) from exc

        if array.ndim == 1:
            tuple_size = 1
        elif array.ndim == 2 and array.shape[1] in SUPPORTED_TUPLE_SIZES:
            tuple_size = int(array.shape[1])
        else:
            raise ValueError(
                "out[%r] has shape %r; expected (%d,) for a float attribute "
                "or (%d, 3) for a vector attribute"
                % (name, array.shape, npoints, npoints)
            )

        if array.shape[0] != npoints:
            raise ValueError(
                "out[%r] has %d rows but the geometry has %d points"
                % (name, array.shape[0], npoints)
            )

        results.append((name.encode("utf-8"), tuple_size, array.tobytes()))

    return results


# Per-node scratch that survives between cooks.
#
# The Warp Code's globals are rebuilt on every cook, so `globals()[...] = x`
# does not persist, so anything cached there is recomputed on every
# frame. That matters: on a 150x150 grid, rebuilding the neighbour graph each
# frame costs ~600ms while the simulation itself costs ~24ms. `store` gives
# the field somewhere to keep expensive, topology-derived data across cooks,
# which is what a Python SOP gets for free by stashing it in an imported
# module.
#
# Keyed by node path, so two nodes never share. Nothing evicts it; it is
# scratch for the session, and a field that keys its entries on something
# topology-derived (point count, prim count) recomputes when that changes.
_NODE_STORE = {}


def node_store(node_path):
    """Return the persistent per-node dict for *node_path*."""
    return _NODE_STORE.setdefault(node_path, {})


def kernels_in_module(module):
    """Return every wp.Kernel in *module*, keyed by its own name.

    The field is ordinary Python and may define several kernels; binding each
    under the name it was written with means `wp.launch(rd_step, ...)` works
    exactly as it reads. The kernel named in Kernel Name is additionally bound
    as `kernel`.
    """
    import warp as wp

    found = {}
    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        if isinstance(value, wp.Kernel):
            if name in RESERVED_NAMES and name != "kernel":
                raise ValueError(
                    "kernel %r collides with a name the node provides (%s); "
                    "rename the kernel"
                    % (name, ", ".join(sorted(RESERVED_NAMES)))
                )
            found[name] = value
    return found


class _HoudiniGeoAdoption:
    """Keeps a writable geometry and its locked GU_DetailHandle alive.

    C++ reads ``address`` and calls ``GA_Detail::replaceWith()`` on that
    pointer. The pointer is only valid while the handle is held, so this
    object owns both and must outlive the C++ call. This is the protocol
    inlinecpp uses (it keeps the handle wrapper for the duration of the call
    and destroys it afterwards).

    cookMySop holds a reference across replaceWith() and drops it after, at
    which point the handle is released.
    """

    def __init__(self, geometry):
        self.geometry = geometry
        self.handle = geometry._guDetailHandle()
        if self.handle.isReadOnly():
            self.handle.destroy()
            self.handle = None
            raise ValueError(
                "houdiniGeo is unexpectedly read-only; cannot adopt it"
            )
        # SwigPyObject wrapping void*. int() yields the address; ctypes.cast
        # does not work on it and __long__ does not exist.
        self.address = int(self.handle._asVoidPointer())

    def __del__(self):
        # C++ holds the only other reference and drops it once replaceWith()
        # has returned, so this runs when the cook is finished with the
        # pointer. __del__ must not raise.
        try:
            if self.handle is not None:
                self.handle.destroy()
                self.handle = None
        except Exception:
            pass


class _WarpNamespace(dict):
    """Globals for the Warp Code field, with a lazily-created ``houdiniGeo``.

    ``houdiniGeo`` has to be a REAL ``hou.Geometry``: a forwarding proxy fails
    ``isinstance(houdiniGeo, hou.Geometry)`` and anything that type-checks its
    argument, so the whole HOM geometry API is not genuinely available through
    one. But materializing it costs a full copy of the input, which code that
    never mentions it should not pay for.

    exec() resolves a missing global through ``__missing__`` on a dict
    subclass, for module-level code and for functions defined in the field.
    The copy therefore happens on first mention of the name and not otherwise. A KeyError from here surfaces to the user as a normal
    NameError.
    """

    def __init__(self, initial, input_geo):
        super().__init__(initial)
        self._input_geo = input_geo
        self.houdini_geo = None

    def __missing__(self, key):
        if key != "houdiniGeo":
            raise KeyError(key)

        # clone_data_ids MUST stay False. With cloned ids the copy keeps the
        # input's data ids. replaceWith() skips attributes whose ids match,
        # so edits to attributes that already existed on the input would be
        # dropped without an error. Newly added attributes are unaffected,
        # which makes the failure hard to spot.
        self.houdini_geo = self._input_geo.freeze(read_only=False)
        self["houdiniGeo"] = self.houdini_geo
        return self.houdini_geo


def _build_namespace(module, kernel, geo, npoints, device, out_attrib,
                     node_path):
    """Build the globals the Warp Code field runs in.

    The names are documented at the top of this module. ``houdiniGeo`` is
    deliberately absent: _WarpNamespace supplies it on first mention, so a
    field that never uses it pays no geometry copy.
    """
    import hou
    import numpy
    import warp as wp

    namespace = _WarpNamespace({
        "wp": wp,
        "np": numpy,
        "hou": hou,
        "kernel": kernel,
        "geo": geo,
        "npoints": npoints,
        "device": device,
        "out_attrib": out_attrib,
        "store": node_store(node_path),
        "point_attrib_array": lambda name: point_attrib_array(geo, name),
        # Seeded so that a field which never assigns `out` gets the
        # explanatory error from marshal_results() rather than a KeyError.
        "out": None,
    }, geo)

    # Each kernel under the name it was written with, so wp.launch(rd_step,
    # ...) reads as it works. Collisions with the names above were already
    # refused, so nothing here can be shadowed.
    namespace.update(kernels_in_module(module))
    return namespace


def run_warp_code(
    kernel_source, kernel_name, warp_code, device, npoints, out_attrib,
    node_path,
):
    """Entry point called by ``SOP_NvidiaWarp::cookMySop``.

    Compiles the Kernel field, builds the namespace documented at the top of
    this module, runs the Warp Code field in it and marshals ``out`` back for
    C++ to write onto the geometry.

    Returns ``(results, adoption)``. *adoption* is None unless the field
    touched ``houdiniGeo``, in which case it is a _HoudiniGeoAdoption whose
    ``address`` C++ passes to replaceWith(). The caller MUST keep it alive
    for the duration of that call.
    """
    import hou
    import numpy
    import warp as wp

    wp.init()

    if device == "cuda" and not wp.get_cuda_device_count():
        raise ValueError("CUDA device requested but none is available")

    module = import_kernel_module(kernel_source, kernel_name)
    kernel = getattr(module, kernel_name, None)
    if kernel is None:
        raise ValueError(
            "kernel source ran but did not define %r" % (kernel_name,)
        )

    geo = input_geometry(node_path)
    namespace = _build_namespace(
        module, kernel, geo, npoints, device, out_attrib, node_path)

    exec_warp_code(warp_code, namespace)

    out = namespace.get("out")

    if namespace.houdini_geo is None:
        # No copy was made, so `out` is the only output and is mandatory.
        return marshal_results(out, npoints, out_attrib), None

    adoption = _HoudiniGeoAdoption(namespace.houdini_geo)

    # With houdiniGeo adopted, the geometry itself is the result, so `out` is
    # optional and layers extra attributes on top. Its arrays are sized
    # against the ADOPTED geometry, whose point count may differ from the
    # input if the field changed topology.
    if out is None:
        return [], adoption

    adopted_npoints = len(adoption.geometry.points())
    return marshal_results(out, adopted_npoints, out_attrib), adoption
