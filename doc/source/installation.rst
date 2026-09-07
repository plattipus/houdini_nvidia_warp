Installation
============

Requirements
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Version
   * - Houdini
     - 20.0, which ships Python 3.10
   * - warp-lang
     - 1.17 or later, below 2.0
   * - rez
     - 3.x
   * - Compiler
     - whatever ``hcustom`` uses on the platform

Houdini as a rez package
------------------------

The package depends on Houdini through a rez variant, so Houdini itself must
be available as an external rez package that points at the vendor install and
exports ``HFS``:

.. code-block:: python

   name = "houdini"
   version = "20.0.506"

   requires = ["platform-osx", "arch-arm64"]

   def commands():
       hfs = "/path/to/Houdini/Resources"
       env.HFS = hfs
       env.PATH.append(hfs + "/bin")

Downstream packages then depend on ``houdini-20.0`` rather than hardcoding an
absolute path.

Building
--------

.. code-block:: bash

   rez build            # compile into ./build
   rez build --install  # install into the local rez package repository

``rez build`` invokes ``build.py``, which calls ``hcustom`` on
``src/SOP_NvidiaWarp.C`` and copies the resulting dso, the ``python/`` tree and
the node icon into the package.

Variants
--------

Only ``houdini-20.0`` is declared, because that is the series the package is
built and tested against. Houdini 19.5 ships Python 3.9, which cannot resolve a
``python-3.10`` ``warp_lang`` variant. Add a variant only alongside a real
build of that series.

Environment
-----------

``commands()`` prepends the package to two search paths and preserves
Houdini's own entries by seeding ``&`` when the variable is unset:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Purpose
   * - ``HOUDINI_DSO_PATH``
     - locates the compiled operator
   * - ``HOUDINI_PATH``
     - locates ``config/Icons`` for the node icon
   * - ``PYTHONPATH``
     - locates the ``plattipus_nvidia_warp`` module

Running
-------

.. code-block:: bash

   rez env plattipus_nvidia_warp -- houdini
