"""End-to-end check that the NVIDIA Warp SOP works inside Houdini.

Builds a small scene with the hou API, runs a Warp kernel through the node and
checks the result. Run it with a Houdini that can see the package:

    rez env plattipus_nvidia_warp -- hython tests/houdini_smoke.py

Exits non-zero if any check fails. Logic that does not need Houdini is covered
by tests/test_kernel_utils.py, which runs under plain pytest.
"""

import sys

import hou

FAILURES = []


def check(label, got, want):
    ok = got == want
    print("%-46s %s" % (label, "PASS" if ok else "FAIL (%r != %r)" % (got, want)))
    if not ok:
        FAILURES.append(label)


def close(label, got, want, tol=1e-4):
    ok = len(got) == len(want) and all(abs(a - b) < tol for a, b in zip(got, want))
    print("%-46s %s" % (label, "PASS" if ok else "FAIL (%r != %r)" % (got, want)))
    if not ok:
        FAILURES.append(label)


def cook(node, kernel=None, kernel_name=None, warp_code=None):
    """Set the fields that were given and return the cooked geometry."""
    if kernel is not None:
        node.parm("kernel").set(kernel)
    if kernel_name is not None:
        node.parm("kernelname").set(kernel_name)
    if warp_code is not None:
        node.parm("warpcode").set(warp_code)
    try:
        node.cook(force=True)
    except hou.Error:
        pass
    return node.geometry()


def error_tail(node):
    return node.errors()[0].strip().splitlines()[-1] if node.errors() else ""


# A kernel that writes the distance of each point from the origin.
LENGTH_KERNEL = """import warp as wp


@wp.kernel
def measure(points: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = wp.length(points[i])
"""

LAUNCH = """P = point_attrib_array("P")

with wp.ScopedDevice(device):
    points = wp.array(P, dtype=wp.vec3)
    result = wp.zeros(npoints, dtype=float)
    wp.launch(measure, dim=npoints, inputs=[points], outputs=[result])
%s
"""


geo = hou.node("/obj").createNode("geo")
box = geo.createNode("box")
node = geo.createNode("plattipus::nvidia_warp::1.0")
node.setFirstInput(box)


# The operator registered, with the icon the package ships.
check("operator name", node.type().name(), "plattipus::nvidia_warp::1.0")
check("operator label", node.type().description(), "NVIDIA Warp")
check("node icon", node.type().icon(), "plattipus_nvidia_warp")

# Parameter order is part of saved-file compatibility.
check("parameters", [p.name() for p in node.parms()],
      ["tabs1", "kernel", "kernelname", "warpcode", "device", "outattrib",
       "enable"])

# The shipped defaults deform the geometry, so a new node does something.
default = [tuple(p.position()) for p in node.geometry().points()]
source = [tuple(p.position()) for p in box.geometry().points()]
check("defaults produce a result", default != source, True)
check("defaults cook cleanly", node.errors(), ())

# A kernel launched by the name the Kernel field gave it, writing one float
# attribute per point. Unit box corners are sqrt(0.75) from the origin.
out = cook(node, LENGTH_KERNEL, "measure", LAUNCH % "    out = result")
close("kernel result written to Output Attribute",
      [p.attribValue("warp_out") for p in out.points()], [0.8660254] * 8)

# A dict names its own attributes and (npoints, 3) becomes a vector.
out = cook(node, warp_code=LAUNCH % '    out = {"len": result, "double": P * 2.0}')
check("dict writes a float attribute", out.findPointAttrib("len").size(), 1)
check("dict writes a vector attribute", out.findPointAttrib("double").size(), 3)

# Row i must reach point i. A Blast leaves gaps in the point offsets, so
# offset order and point order differ and a wrong mapping shows up here.
blast = geo.createNode("blast")
blast.setFirstInput(box)
blast.parm("group").set("0-2")
node.setFirstInput(blast)
out = cook(node, warp_code='out = {"ptnum": np.arange(npoints, dtype=np.float32)}')
close("row i is written to point i",
      [p.attribValue("ptnum") for p in out.points()],
      [float(i) for i in range(len(out.points()))])
node.setFirstInput(box)

# houdiniGeo is a real hou.Geometry and may change topology.
out = cook(node, warp_code="p = houdiniGeo.createPoint()\n"
                           "p.setPosition(hou.Vector3(0, 5, 0))\n")
check("houdiniGeo adds a point", len(out.points()), 9)

out = cook(node, warp_code='houdiniGeo.setPointFloatAttribValuesFromString(\n'
                           '    "P", np.zeros(npoints * 3, np.float32),\n'
                           '    hou.numericData.Float32)\n')
check("houdiniGeo edits an existing attribute",
      all(tuple(p.position()) == (0.0, 0.0, 0.0) for p in out.points()), True)

# store survives between cooks; globals() does not.
cook(node, warp_code="store['n'] = store.get('n', 0) + 1\n"
                     "out = {'n': np.full(npoints, float(store['n']), np.float32)}\n")
first = node.geometry().points()[0].attribValue("n")
node.parm("outattrib").set("dirty")
node.cook(force=True)
check("store persists between cooks",
      node.geometry().points()[0].attribValue("n") > first, True)
node.parm("outattrib").revertToDefaults()

# Failures land on the node instead of crashing Houdini.
cook(node, kernel="this is not python (")
check("kernel syntax error is reported", "syntax error" in error_tail(node), True)

cook(node, kernel=LENGTH_KERNEL, kernel_name="measure",
     warp_code="this is not python (")
check("Warp Code syntax error is reported",
      "syntax error" in error_tail(node), True)

cook(node, warp_code="x = 1\n")
check("unassigned out is reported", "`out`" in error_tail(node), True)

# ... and the node recovers afterwards.
out = cook(node, warp_code=LAUNCH % "    out = result")
check("node recovers after an error", node.errors(), ())
check("geometry survives an error", len(out.points()), 8)

# Disabling the node passes the input through.
node.parm("enable").set(0)
out = cook(node)
check("disabled node passes input through", len(out.points()), 8)
check("disabled node writes nothing", out.findPointAttrib("warp_out"), None)


print()
if FAILURES:
    print("%d FAILED: %s" % (len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("all checks passed")
