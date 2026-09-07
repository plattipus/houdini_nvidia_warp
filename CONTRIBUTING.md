# Contributing

Thanks for your interest. Bug reports and pull requests are welcome.

## Reporting a bug

Include the Houdini version, the Warp version, the platform, and the smallest
Kernel and Warp Code that reproduce the problem. If the node reported an error,
paste it in full: errors from either code field carry the line number inside
that field.

## Building

```bash
rez build            # compile into ./build
rez build --install  # install into the local rez package repository
```

`rez build` runs `build.py`, which calls `hcustom` on `src/SOP_NvidiaWarp.C`.
The build must compile with no warnings; `hcustom` enables `-Wall -W`.

## Testing

```bash
rez env plattipus_nvidia_warp -- hython tests/test_nvidia_warp.py
```

The test builds a scene with the `hou` API, runs kernels through the node and
checks the results. It needs Houdini and exits non-zero on failure. Please run
it before opening a pull request, and add a check for anything you fix or add.

## Documentation

```bash
pip install sphinx furo
sphinx-build -b html doc/source doc/build -W
```

`-W` turns warnings into errors, and CI builds the documentation this way.

Note that `automodule` only documents names listed in `__all__`, so a new
public function needs adding there as well as to `doc/source/api.rst`.

## Style

- Python follows PEP 8. C++ follows the surrounding HDK conventions.
- Comments explain why, not what. Leave out the history of how the code got
  to its current shape.
- Every `PyObject*` needs a matching decref on every path. The C++ uses
  `PY_AutoObject` for this; keep to it, and remember that `PyList_SetItem` and
  `PyTuple_SetItem` steal a reference and must not be wrapped.
- Inserting, removing or reordering a `PRM_Template` shifts parameter indices
  and silently remaps values in saved `.hip` files. Append new parameters to
  the end of their tab, and say so in the pull request.

## Releasing

1. Update `version` in `package.py` following semantic versioning.
2. Move the `## Unreleased` section of `CHANGELOG.md` under a new heading.
3. Run the test and build the documentation.
4. `rez build --install`, then tag the commit `v<version>`.
