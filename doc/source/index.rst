plattipus_nvidia_warp
=====================

**Executes an NVIDIA Warp kernel on geometry.**

The NVIDIA Warp SOP provides a general interface to write and run
`NVIDIA Warp <https://github.com/NVIDIA/warp>`_ kernels on geometry inside
`Houdini <https://www.sidefx.com>`_. Warp compiles Python to native CPU and
CUDA code, so a kernel written on this node runs at native speed.

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
