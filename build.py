# -*- coding: utf-8 -*-
"""rez build_command for plattipus_nvidia_warp.

Deliberately thin: it shells out to `hcustom` and then moves the resulting
dso and the python/ tree into place. No compilation logic lives here. If a
compiler flag needs changing, it belongs in the hcustom invocation, not in a
hand-rolled build system.
"""

from __future__ import print_function

import os
import shutil
import subprocess
import sys


SOURCE = os.path.join('src', 'SOP_NvidiaWarp.C')


def _env(name):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            'error: %s is not set. This script must be run by `rez build`, '
            'not directly.' % name
        )
    return value


def main():
    install = 'install' in sys.argv[1:]

    source_path = _env('REZ_BUILD_SOURCE_PATH')
    build_path = _env('REZ_BUILD_PATH')

    hfs = os.environ.get('HFS')
    if not hfs:
        raise SystemExit(
            'error: HFS is not set. The houdini variant did not resolve, so '
            'hcustom cannot be located. Check `requires`/`variants` in '
            'package.py.'
        )

    src = os.path.join(source_path, SOURCE)
    if not os.path.isfile(src):
        raise SystemExit('error: missing source file: %s' % src)

    dso_dir = os.path.join(build_path, 'dso')
    if not os.path.isdir(dso_dir):
        os.makedirs(dso_dir)

    hcustom = os.path.join(hfs, 'bin', 'hcustom')
    if not os.path.isfile(hcustom):
        raise SystemExit('error: hcustom not found at %s' % hcustom)

    # -e echoes the underlying compiler/linker command so build logs show the
    # real invocation rather than just hcustom's summary.
    cmd = [hcustom, '-e', '-i', dso_dir, src]

    print('+ ' + ' '.join(cmd), flush=True)
    result = subprocess.call(cmd, cwd=source_path)
    if result != 0:
        raise SystemExit('error: hcustom failed with exit code %d' % result)

    # hcustom drops the intermediate object file next to the source. Remove
    # it so a build never leaves artifacts in the source tree.
    stale_obj = os.path.splitext(src)[0] + '.o'
    if os.path.isfile(stale_obj):
        os.remove(stale_obj)

    # The node icon ships with the package; package.py puts this
    # directory on HOUDINI_PATH so Houdini finds config/Icons/.
    cfg_src = os.path.join(source_path, 'config')
    cfg_dst = os.path.join(build_path, 'config')
    if os.path.isdir(cfg_src):
        if os.path.isdir(cfg_dst):
            shutil.rmtree(cfg_dst)
        shutil.copytree(cfg_src, cfg_dst)

    # The sample python module tree ships alongside the dso.
    py_src = os.path.join(source_path, 'python')
    py_dst = os.path.join(build_path, 'python')
    if os.path.isdir(py_src):
        if os.path.isdir(py_dst):
            shutil.rmtree(py_dst)
        shutil.copytree(
            py_src, py_dst,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'),
        )

    if install:
        install_path = _env('REZ_BUILD_INSTALL_PATH')
        for name in ('dso', 'python', 'config'):
            src_dir = os.path.join(build_path, name)
            if not os.path.isdir(src_dir):
                continue
            dst_dir = os.path.join(install_path, name)
            if os.path.isdir(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            print('installed %s -> %s' % (src_dir, dst_dir))

    return 0


if __name__ == '__main__':
    sys.exit(main())
