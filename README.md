<img src="doc/source/_static/logo.svg" alt="" width="112" align="right">

# plattipus_nvidia_warp

A compiled [Houdini](https://www.sidefx.com) SOP that runs
[NVIDIA Warp](https://github.com/NVIDIA/warp) kernels over geometry.

Warp compiles Python to native CPU and CUDA code. This node lets you write a
kernel and the host code that launches it directly in Houdini, with the
geometry already in scope, instead of maintaining a Python SOP that reimports
your modules on every cook.

By [plattipus.com](https://plattipus.com).

```python
# Kernel tab                        # Warp Code tab
import warp as wp                   P = point_attrib_array("P")

@wp.kernel                          with wp.ScopedDevice(device):
def measure(                            points = wp.array(P, dtype=wp.vec3)
    p: wp.array(dtype=wp.vec3),         result = wp.zeros(npoints, dtype=float)
    out: wp.array(dtype=float)):        wp.launch(measure, dim=npoints,
    i = wp.tid()                                  inputs=[points],
    out[i] = wp.length(p[i])                      outputs=[result])

                                    out = result
```

It is a real compiled operator, not a scripted node: every parameter is a
`PRM_Template`, so the node has no Python startup cost and no `hou.session`
dependencies.

---

## Contents

- [Requirements](#requirements)
- [Install](#install)
- [The three tabs](#the-three-tabs)
- [Names available to the Warp Code](#names-available-to-the-warp-code)
- [Returning results](#returning-results)
- [Writing geometry directly](#writing-geometry-directly)
- [State between cooks](#state-between-cooks)
- [Simulations](#simulations)
- [Performance](#performance)
- [Limitations](#limitations)
- [Development](#development)

## Requirements

| | |
|---|---|
| Houdini | 20.0 (Python 3.10) |
| warp-lang | `>=1.17,<2` |
| rez | 3.x |
| Compiler | whatever `hcustom` uses on the platform |

Houdini must be available as an external rez package exporting `HFS`. See
[the installation guide](doc/source/installation.rst).

## Install

```bash
rez build --install
rez env plattipus_nvidia_warp -- houdini
```

Drop an **NVIDIA Warp** SOP after any geometry. The shipped defaults are a
ripple deformer, so a new node produces visible output immediately.

## The three tabs

**Kernel** holds `@wp.kernel` device code. Every kernel defined here is
available to the Warp Code under the name you gave it, so
`wp.launch(my_kernel, ...)` reads as it works. The one named in **Kernel Name**
is also bound as `kernel`.

**Warp Code** is host-side Python, run once per cook. It reads geometry, builds
`wp.array`s, launches kernels and returns results.

**Options** sets the device (`cpu` or `cuda`), the default output attribute
name, and an enable toggle.

Both code fields are multi-line Python editors.

## Names available to the Warp Code

| Name | Description |
|---|---|
| `wp`, `np`, `hou` | warp, numpy, and the Houdini Python module |
| `kernel` | the kernel named in **Kernel Name** |
| `geo` | read-only input geometry, a `hou.Geometry` |
| `houdiniGeo` | writable geometry, created on first use |
| `npoints` | point count of the input |
| `device` | `"cpu"` or `"cuda"` |
| `out_attrib` | the **Output Attribute** name |
| `point_attrib_array(name)` | float32 numpy view of a point attribute |
| `store` | dict that persists between cooks |
| `out` | assign results here |

## Returning results

```python
out = {"name": array, ...}   # each key names a point attribute
out = array                  # shorthand for {out_attrib: array}
out = {}                     # write nothing
```

An array shaped `(npoints,)` becomes a float attribute; `(npoints, 3)` becomes
a vector. Values may be numpy arrays, `wp.array` objects, or plain sequences of
floats.

Leaving `out` unassigned is an error, so a typo cannot produce a node that
cooks cleanly and writes nothing.

## Writing geometry directly

`houdiniGeo` is a writable `hou.Geometry` supporting the whole HOM API,
including topology changes:

```python
houdiniGeo.setPointFloatAttribValuesFromString(
    "P", positions, hou.numericData.Float32)

p = houdiniGeo.createPoint()
p.setPosition(hou.Vector3(0, 1, 0))
```

It copies the input the first time you mention it, and the node then adopts it
as the cook result. A field that never mentions it makes no copy, so prefer
`point_attrib_array` and `out` when you only need to write attributes.

## State between cooks

The Warp Code namespace is rebuilt on every cook, so `globals()` does not
persist. Use `store` for anything expensive that depends only on topology:

```python
key = (npoints, geo.intrinsicValue("primitivecount"))
if key not in store:
    store[key] = build_connectivity(geo)
offsets, neighbours = store[key]
```

Keying on topology-derived values means the entry rebuilds when the geometry
changes.

## Simulations

The node cooks from its input every time, so state does not accumulate on its
own. Wrap it in a Solver SOP and wire `Prev_Frame` to its input.

`examples/gray_scott.hip` is a Gray-Scott reaction-diffusion set up this way,
with both code fields also saved as `.py` files beside it.

## Performance

Measured on a 300x300 grid, 90,000 points and 89,401 primitives, one
reaction-diffusion step per cook:

| Approach | Cook time |
|---|---|
| Python SOP doing the same work | 155.6 ms |
| This node, via `houdiniGeo` | 145.9 ms |
| This node, via `geo` and `out` | 96.0 ms |

Two things dominate real workloads, and neither is the node:

**Rebuilding topology data every frame.** On a 150x150 grid a Python neighbour
graph costs about 618 ms per frame against a 24 ms simulation. Caching it in
`store` brings the frame down to about 7 ms.

**Per-element attribute access.** `point_attrib_array` and
`setPointFloatAttribValuesFromString` move whole buffers; the equivalent
Python-list calls are roughly 18 times slower.

## Limitations

- **`hou.pwd().geometry()` does not work**, here or in any node's own
  parameters. It asks the node to cook while it is already cooking, which
  Houdini reports as *Infinite recursion in evaluation*. Use `geo` to read and
  `houdiniGeo` to write. A Python SOP is the exception because Houdini hands it
  the working geometry through a layer that is not in the public HDK.
- One input.
- `out` writes float and vector **point** attributes. Other classes and types
  must go through `houdiniGeo`.
- Only the `houdini-20.0` variant is built and tested. Houdini 19.5 ships
  Python 3.9, which cannot resolve a `python-3.10` `warp_lang`.
- The CUDA path is implemented but untested on the development machine, whose
  Warp build reports `CUDA not enabled`.

## Development

```bash
rez build                                                   # compile
rez env plattipus_nvidia_warp -- hython tests/test_nvidia_warp.py
pip install sphinx furo && sphinx-build -b html doc/source doc/build -W
```

The test builds a scene with the `hou` API, runs kernels through the node and
checks the results. It needs Houdini and exits non-zero on failure. `rez build`
does not build the documentation.

Full documentation is in [`doc/source`](doc/source).

## Contributing

Bug reports and pull requests are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for how to build, test and submit changes.

## Credits

Built by [plattipus](https://plattipus.com).

This project is an independent integration and is not affiliated with or
endorsed by either of the projects it builds on:

- [Houdini](https://www.sidefx.com) and the HDK, by
  [SideFX](https://www.sidefx.com).
- [NVIDIA Warp](https://github.com/NVIDIA/warp), by
  [NVIDIA](https://www.nvidia.com), under the Apache 2.0 licence.

"NVIDIA" and "Warp" are trademarks of NVIDIA Corporation. "Houdini" is a
trademark of Side Effects Software Inc. Both are used here only to describe
what this node interoperates with.

## License

MIT. See [LICENSE](LICENSE).
