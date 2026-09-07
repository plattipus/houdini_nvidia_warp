Python API
==========

The ``plattipus_nvidia_warp.kernel_utils`` module holds the Python that the compiled
SOP calls into. The Warp Code field does not import it directly; the node
builds the field's namespace from it.

.. automodule:: plattipus_nvidia_warp.kernel_utils
   :members: validate_kernel, import_kernel_module,
             kernels_in_module, input_geometry, point_attrib_array,
             exec_warp_code, marshal_results, node_store, run_warp_code
   :member-order: bysource
