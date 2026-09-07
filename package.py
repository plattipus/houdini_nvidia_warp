# -*- coding: utf-8 -*-

name = 'plattipus_nvidia_warp'

version = '1.0.0'

description = \
    """
    Executes an NVIDIA Warp kernel on geometry.

    A compiled HDK SOP, plattipus::nvidia_warp::1.0, labelled "NVIDIA Warp".

    The node is structured like Houdini's own Python/OpenCL SOPs: a Kernel
    tab for the @wp.kernel device code, a Warp Code tab for the host-side
    Python that reads geometry and launches the kernel, and an Options tab
    all as real PRM_Template entries, not a scripted node with a raw
    Python string parm.
    """

authors = ['plattipus']

# warp_lang is a runtime dependency: the compiled SOP imports it through the
# embedded interpreter at cook time. Houdini itself is expressed as a variant
# below rather than here, so each supported Houdini series gets its own
# separately-compiled dso (the HDK is not ABI-stable across releases).
requires = [
    'warp_lang-1.17+<2',
]

# Only houdini-20.0 is listed because it is the only series actually built and
# tested here. Houdini 19.5 ships Python 3.9, and the available warp_lang
# variant is python-3.10 only, so a 19.5 variant could not resolve warp_lang.
# Add a variant here only alongside a real build of that series.
variants = [
    ['houdini-20.0'],
]

build_command = 'python {root}/build.py {install}'

uuid = 'plattipus.nvidia_warp'


def commands():
    # The sample kernel-helper module the SOP imports at cook time.
    env.PYTHONPATH.append('{root}/python')

    # HOUDINI_DSO_PATH is a search path, and Houdini only searches its own
    # default locations if '&' appears in it. If nothing upstream has set the
    # variable, seed it with '&' first so prepending our dso dir extends the
    # default path instead of replacing it. Otherwise every other plugin in
    # the session silently stops loading.
    if not defined('HOUDINI_DSO_PATH'):
        env.HOUDINI_DSO_PATH = '&'
    env.HOUDINI_DSO_PATH.prepend('{root}/dso')

    # The node icon lives in config/Icons. HOUDINI_PATH needs the same '&'
    # treatment as HOUDINI_DSO_PATH: without it Houdini stops searching its
    # own directories and the session loses far more than an icon.
    if not defined('HOUDINI_PATH'):
        env.HOUDINI_PATH = '&'
    env.HOUDINI_PATH.prepend('{root}')
