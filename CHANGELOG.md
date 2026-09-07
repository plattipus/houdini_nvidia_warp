# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses [semantic versioning](https://semver.org/).

## Unreleased

## 1.0.0 - 2026-09-07

First public release.

### Added

- Compiled SOP `plattipus::nvidia_warp::1.0`, labelled "NVIDIA Warp", with
  Kernel, Warp Code and Options tabs as compiled `PRM_Template` entries.
- Kernel field compiled through a temporary module and cached on a hash of the
  source. Every kernel it defines is bound in the Warp Code under its own name.
- Warp Code field with `wp`, `np`, `hou`, `kernel`, `geo`, `houdiniGeo`,
  `npoints`, `device`, `out_attrib`, `point_attrib_array()` and `store`.
- `out` contract for writing point attributes, supporting a dict of arrays, a
  bare array, or an empty dict.
- `houdiniGeo`, a writable `hou.Geometry` created on first use, adopted as the
  cook result through `GA_Detail::replaceWith()`. Supports topology changes.
- `store`, a per-node dict that persists between cooks.
- Zero-copy attribute reads through `pointFloatAttribValuesAsString`.
- rez package with a `houdini-20.0` variant, node icon, and `HOUDINI_DSO_PATH`
  and `HOUDINI_PATH` wiring that preserves Houdini's default search paths.
- Gray-Scott reaction-diffusion examples.
- Test suites: pytest for the Python helpers, and a Houdini smoke test.
