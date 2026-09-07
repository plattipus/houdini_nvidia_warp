plattipus_nvidia_warp
=====================

A compiled Houdini SOP that runs `NVIDIA Warp <https://github.com/NVIDIA/warp>`_
kernels over geometry.

The node is structured like Houdini's own Python and OpenCL SOPs: a **Kernel**
tab for the ``@wp.kernel`` device code, a **Warp Code** tab for the host-side
Python that reads geometry and launches the kernel, and an **Options** tab.
All parameters are compiled ``PRM_Template`` entries.

.. list-table::
   :widths: 30 70

   * - Operator
     - ``plattipus::nvidia_warp::1.0``
   * - Label
     - NVIDIA Warp
   * - Inputs
     - 1

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   performance
   api

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
