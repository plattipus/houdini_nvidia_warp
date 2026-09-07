/*
 * SOP_NvidiaWarp.C
 *
 * Operator plattipus::nvidia_warp::1.0, labelled "NVIDIA Warp".
 *
 * A compiled SOP that evaluates an NVIDIA Warp kernel over the input
 * geometry. Structured like Houdini's own Python / OpenCL SOPs: two large
 * Python fields on their own tabs plus an Options tab, all as real
 * PRM_Template entries. There is deliberately no "script" parm and no solver
 * registry.
 *
 *   Kernel tab      the @wp.kernel device code
 *   Warp Code tab   the host-side Python that reads geometry, builds
 *                   wp.arrays, calls wp.launch() and produces results
 *   Options tab     device, output attribute name, enable
 *
 * Data flow:
 *
 *   read   The Warp Code field reads geometry itself, through HOM, with
 *          hou.node(<this node>).inputs()[0].geometry() and a bulk numpy
 *          read of each attribute. No geometry is marshalled from C++.
 *
 *   write  Results come back to C++ and are written with GA_RWHandleF /
 *          GA_RWHandleV3. HOMF, the GU_Detail to HOM_Geometry bridge, is
 *          not shipped in the public toolkit, so
 *          there is no writable hou.Geometry for the cooking detail, and
 *          calling .geometry() on this node during its own cook raises
 *          "Infinite recursion in evaluation". Do not try to "simplify" this
 *          by writing from Python; there is no public API for it.
 *
 * Python is reached through Houdini's own embedded interpreter via
 * PY_InterpreterAutoLock and the PY_-prefixed CPython wrappers in
 * PY_CPythonAPI.h. Every new reference is owned by a PY_AutoObject so that
 * refcounts stay correct on every early-return path.
 */

// PY_CPythonAPI.h stands in for Python.h and must come before other headers.
#include <PY/PY_CPythonAPI.h>
#include <PY/PY_AutoObject.h>
#include <PY/PY_InterpreterAutoLock.h>
#include <PY/PY_Python.h>
#include <PY/PY_Result.h>

#include <UT/UT_DSOVersion.h>
#include <UT/UT_String.h>
#include <UT/UT_StringHolder.h>
#include <UT/UT_Vector3.h>
#include <UT/UT_WorkBuffer.h>

#include <SOP/SOP_Node.h>
#include <OP/OP_AutoLockInputs.h>
#include <OP/OP_Operator.h>
#include <OP/OP_OperatorTable.h>
#include <PRM/PRM_Include.h>
#include <PRM/PRM_SpareData.h>
#include <GU/GU_Detail.h>
#include <GA/GA_Attribute.h>
#include <GA/GA_Handle.h>
#include <GA/GA_Types.h>

#include <string.h>

namespace
{

// The Python module the node drives. Shipped as python/plattipus_nvidia_warp/ in
// this package and put on PYTHONPATH by package.py's commands().
const char *const theWarpModule = "plattipus_nvidia_warp.kernel_utils";
const char *const theWarpEntryPoint = "run_warp_code";

// ---------------------------------------------------------------------------
// Parameters
//
// WARNING: the order of entries in theTemplateList below is part of this
// node's saved-file compatibility. Houdini stores parm values by index, so
// inserting, removing or reordering an entry will silently re-map parameter
// values in existing .hip files. Append new parms at the end of their tab and
// bump the operator version if the order must ever change.
// ---------------------------------------------------------------------------

PRM_Name theSwitcherName("tabs", "Tabs");
PRM_Default theSwitcherDefaults[] = {
    PRM_Default(2, "Kernel"),
    PRM_Default(1, "Warp Code"),
    PRM_Default(3, "Options"),
};

const char *const theKernelHelp =
    "NVIDIA Warp device code. Must define the function named by Kernel Name "
    "and decorate it with @wp.kernel.\n"
    "\n"
    "This source is written to a file and imported, not exec()'d: Warp's code "
    "generator calls inspect.getsourcelines() on the decorated function and "
    "rejects kernels defined as a string.\n"
    "\n"
    "The compiled kernel is handed to the Warp Code field as `kernel`.";

// Spare data for the two code fields.
//
// PRM_SpareData::stringEditorLangPython sets ONLY the "editorlang" token --
// it does not set "editor", so on its own the parm renders as a single-line
// text field with no multi-line editor. Houdini's own Python SOP carries all
// three tokens ({'editor': '1', 'editorlang': 'python', 'editorlines':
// '20-50'}), so match that exactly.
PRM_SpareData theCodeEditor(
    PRM_SpareArgs()
        << PRM_SpareToken(PRM_SpareData::getEditorToken(), "1")
        << PRM_SpareToken(PRM_SpareData::getEditorLanguageToken(), "python")
        << PRM_SpareToken(PRM_SpareData::getEditorLinesRangeToken(), "20-50"));

PRM_Name theKernelName("kernel", "Kernel");
PRM_Default theKernelDefault(
    0,
    "import warp as wp\n"
    "\n"
    "\n"
    "@wp.kernel\n"
    "def deform(points: wp.array(dtype=wp.vec3),\n"
    "           amplitude: float,\n"
    "           frequency: float,\n"
    "           out: wp.array(dtype=wp.vec3)):\n"
    "    i = wp.tid()\n"
    "    p = points[i]\n"
    "\n"
    "    # Ripple outwards from the origin in XZ.\n"
    "    r = wp.sqrt(p[0] * p[0] + p[2] * p[2])\n"
    "    p[1] = p[1] + amplitude * wp.sin(r * frequency)\n"
    "\n"
    "    out[i] = p\n");

const char *const theKernelFuncHelp =
    "Name of the @wp.kernel function to launch from the Kernel field.";

PRM_Name theKernelFuncName("kernelname", "Kernel Name");
PRM_Default theKernelFuncDefault(0, "deform");

const char *const theWarpCodeHelp =
    "Host-side Python: read geometry, build wp.arrays, launch the kernel and "
    "return results.\n"
    "\n"
    "Available names:\n"
    "  wp                        the warp module, already wp.init()-ed\n"
    "  np                        numpy\n"
    "  hou                       the hou module\n"
    "  kernel                    the compiled kernel from the Kernel field\n"
    "  geo                       read-only hou.Geometry of input 0\n"
    "  npoints                   point count\n"
    "  device                    the Device parm, 'cpu' or 'cuda'\n"
    "  out_attrib                the Output Attribute parm\n"
    "  point_attrib_array(name)  bulk numpy read of a float point attribute\n"
    "\n"
    "Return results by assigning `out`:\n"
    "  out = {'name': array, ...}  writes one point attribute per key\n"
    "  out = array                 shorthand for {out_attrib: array}\n"
    "  out = {}                    writes nothing; not an error\n"
    "\n"
    "Leaving `out` unassigned is an error, so a typo cannot produce a node "
    "that cooks clean and does nothing.\n"
    "\n"
    "Values may be a numpy array, a wp.array (converted with .numpy()) or "
    "anything convertible to float32. The shape picks the attribute type: "
    "(npoints,) makes a float attribute, (npoints, 3) makes a vector "
    "attribute. Row i is written to point i.\n"
    "\n"
    "geo is READ-ONLY, and it is the *input* geometry. There is no writable "
    "hou.Geometry for this node's own output. Assigning `out` is the only "
    "way to change geometry.";

PRM_Name theWarpCodeName("warpcode", "Warp Code");
PRM_Default theWarpCodeDefault(
    0,
    "# Host-side Warp code. This is ordinary Python.\n"
    "#\n"
    "# houdiniGeo  a writable hou.Geometry, exactly like a Python SOP's:\n"
    "#             the whole HOM API works on it, including topology edits.\n"
    "#             Mentioning it copies the input, so if you only need to\n"
    "#             write attributes, assign `out` instead and never touch it.\n"
    "# geo         the read-only input geometry\n"
    "# kernel      the compiled kernel named in Kernel Name. Every kernel\n"
"#             the Kernel field defines is ALSO bound under its own\n"
"#             name, so wp.launch(my_kernel, ...) works directly\n"
    "# store       a dict that persists between cooks. globals() does not,\n"
"#             so cache expensive topology-derived data here\n"
"# also: wp, np, hou, npoints, device, out_attrib,\n"
    "#       point_attrib_array(name) -> zero-copy float32 array\n"
    "\n"
    "P = point_attrib_array(\"P\")          # (npoints, 3) float32, no copy\n"
    "\n"
    "with wp.ScopedDevice(device):\n"
    "    src = wp.array(P, dtype=wp.vec3)\n"
    "    dst = wp.zeros(npoints, dtype=wp.vec3)\n"
    "\n"
    "    wp.launch(kernel, dim=npoints,\n"
    "              inputs=[src, 0.25, 8.0],   # amplitude, frequency\n"
    "              outputs=[dst])\n"
    "\n"
    "# Bulk write: passes the numpy buffer straight through, which is far\n"
    "# faster than setPointFloatAttribValues() with a Python list.\n"
    "houdiniGeo.setPointFloatAttribValuesFromString(\n"
    "    \"P\", dst.numpy(), hou.numericData.Float32)\n");

const char *const theDeviceHelp =
    "Warp device to launch on. Passed to the Warp Code field as `device`; "
    "the default code wraps the launch in wp.ScopedDevice(device). Selecting "
    "CUDA on a machine with no CUDA device is an error rather than a silent "
    "fallback to CPU.";

PRM_Name theDeviceName("device", "Device");
PRM_Name theDeviceChoices[] = {
    PRM_Name("cpu", "CPU"),
    PRM_Name("cuda", "CUDA"),
    PRM_Name(0)
};
PRM_ChoiceList theDeviceMenu(PRM_CHOICELIST_SINGLE, theDeviceChoices);

const char *const theOutAttribHelp =
    "Default output point attribute name. Passed to the Warp Code field as "
    "`out_attrib`, and used as the attribute name when the field assigns a "
    "bare array to `out` instead of a dict.\n"
    "\n"
    "This is a convenience, not a restriction: Warp Code that assigns a dict "
    "chooses its own attribute names and may write several at once.";

PRM_Name theOutAttribName("outattrib", "Output Attribute");
PRM_Default theOutAttribDefault(0, "warp_out");

const char *const theEnableHelp =
    "When off, the input geometry is passed through untouched and no Python "
    "runs.";

PRM_Name theEnableName("enable", "Enable");

}  // anonymous namespace


class SOP_NvidiaWarp : public SOP_Node
{
public:
    static OP_Node *myConstructor(OP_Network *net, const char *name,
                                  OP_Operator *op)
    {
        return new SOP_NvidiaWarp(net, name, op);
    }

    static PRM_Template myTemplateList[];

protected:
    SOP_NvidiaWarp(OP_Network *net, const char *name, OP_Operator *op)
        : SOP_Node(net, name, op)
    {
        mySopFlags.setManagesDataIDs(true);
    }

    ~SOP_NvidiaWarp() override {}

    OP_ERROR cookMySop(OP_Context &context) override;

private:
    // Pulls the active Python exception (if any) onto the node as an error.
    // Must be called with the GIL held.
    void reportPythonError(const char *what);

    // Writes one (name, tuple_size, data) triple from the Warp Code result
    // onto gdp. Returns false with an error already added to the node.
    // Must be called with the GIL held; borrows *item*, taking no reference.
    bool writeResultItem(PY_PyObject *item, GA_Size npoints);

    // How an empty string parm is reported.
    enum EmptyParm
    {
        PASS_THROUGH,   // the node does nothing; a warning
        MISCONFIGURED,  // the node cannot run; an error
    };

    // Evaluates a string parm, reporting *message* if it is empty. Returns
    // false when the caller should stop cooking.
    bool evalRequiredString(const char *parm, fpreal t, UT_String &value,
                            EmptyParm kind, const char *message);

    // Ensures gdp is populated on every exit path.
    //
    // duplicateSource() must run on any path that returns geometry, but must
    // not run once the Warp Code has adopted houdiniGeo, because replaceWith()
    // has already filled gdp and duplicating would overwrite it. Which case
    // applies is known only after the Warp Code has run, so the copy cannot be
    // made up front.
    //
    // The destructor covers every exit, including early returns. The two paths
    // that populate gdp by other means call duplicateNow() or disarm().
    class SourceGuard
    {
    public:
        SourceGuard(SOP_NvidiaWarp &sop, OP_Context &context)
            : mySop(sop), myContext(context), myArmed(true)
        {}

        // Fires on every path that has not already populated gdp.
        ~SourceGuard() { duplicateNow(); }

        // Populate gdp now, for paths that go on to write onto it.
        void duplicateNow()
        {
            if (myArmed)
            {
                myArmed = false;
                mySop.duplicateSource(0, myContext);
            }
        }

        // gdp was populated another way (replaceWith); do not duplicate.
        void disarm() { myArmed = false; }

        // Non-copyable: a copy would carry myArmed and duplicate the source
        // a second time when the copy died. The whole point of this class is
        // that gdp is populated exactly once.
        SourceGuard(const SourceGuard &) = delete;
        SourceGuard &operator=(const SourceGuard &) = delete;

    private:
        SOP_NvidiaWarp &mySop;
        OP_Context     &myContext;
        bool            myArmed;
    };
};


PRM_Template SOP_NvidiaWarp::myTemplateList[] = {
    PRM_Template(PRM_SWITCHER, 3, &theSwitcherName, theSwitcherDefaults),

    // --- Kernel tab -------------------------------------------------------
    PRM_Template(PRM_STRING, 1, &theKernelName, &theKernelDefault,
                 0, 0, 0, &theCodeEditor,
                 1, theKernelHelp),
    PRM_Template(PRM_STRING, 1, &theKernelFuncName, &theKernelFuncDefault,
                 0, 0, 0, 0, 1, theKernelFuncHelp),

    // --- Warp Code tab ----------------------------------------------------
    PRM_Template(PRM_STRING, 1, &theWarpCodeName, &theWarpCodeDefault,
                 0, 0, 0, &theCodeEditor,
                 1, theWarpCodeHelp),

    // --- Options tab ------------------------------------------------------
    PRM_Template(PRM_ORD, 1, &theDeviceName, PRMzeroDefaults, &theDeviceMenu,
                 0, 0, 0, 1, theDeviceHelp),
    PRM_Template(PRM_STRING, 1, &theOutAttribName, &theOutAttribDefault,
                 0, 0, 0, 0, 1, theOutAttribHelp),
    PRM_Template(PRM_TOGGLE, 1, &theEnableName, PRMoneDefaults,
                 0, 0, 0, 0, 1, theEnableHelp),

    PRM_Template()
};


void
SOP_NvidiaWarp::reportPythonError(const char *what)
{
    // PYextractPythonException() consumes the pending exception, so the
    // interpreter is left with a clean error indicator afterwards.
    PY_Result result = PYextractPythonException();

    const char *detail = "unknown Python error";
    if (result.myDetailedErrValue.isstring())
        detail = result.myDetailedErrValue.c_str();
    else if (result.myErrValue.isstring())
        detail = result.myErrValue.c_str();

    UT_WorkBuffer msg;
    msg.sprintf("%s: %s", what, detail);
    addError(SOP_MESSAGE, msg.buffer());
}


bool
SOP_NvidiaWarp::writeResultItem(PY_PyObject *item, GA_Size npoints)
{
    // kernel_utils.marshal_results() guarantees this shape. Anything else
    // means the sample module and this node have drifted apart, so say so
    // plainly instead of indexing into whatever showed up.
    if (!PY_PySequence_Check(item) || PY_PySequence_Size(item) != 3)
    {
        // PySequence_Size() may have set an error indicator on the way to
        // returning -1. Clear it so it cannot surface at some unrelated later
        // Python call in this session.
        PY_PyErr_Clear();
        addError(SOP_MESSAGE,
                 "kernel_utils.marshal_results() must return "
                 "(name, tuple_size, data) triples.");
        return false;
    }

    // Three new references, each owned by its own PY_AutoObject: they are
    // released by the destructors on every path out of this function,
    // including all the early returns below.
    PY_AutoObject name_obj(PY_PySequence_GetItem(item, 0));
    PY_AutoObject size_obj(PY_PySequence_GetItem(item, 1));
    PY_AutoObject data_obj(PY_PySequence_GetItem(item, 2));
    if (!name_obj.ptr() || !size_obj.ptr() || !data_obj.ptr())
    {
        reportPythonError("Malformed result from the Warp Code field");
        return false;
    }

    // marshal_results() encodes attribute names as UTF-8 bytes precisely so
    // that they can be read here through PyBytes_AsStringAndSize(), which has
    // an unambiguous declared signature, rather than through the PyString_*
    // Python-2 compatibility shim.
    char *name_buf = nullptr;
    PY_Py_ssize_t name_len = 0;
    if (PY_PyBytes_AsStringAndSize(name_obj, &name_buf, &name_len) != 0)
    {
        reportPythonError("Attribute name in the Warp Code result is not bytes");
        return false;
    }

    // A PyBytes buffer always carries a trailing null, so once the payload is
    // known to be null-free, name_buf is a valid C string.
    if (name_len <= 0 || (PY_Py_ssize_t)strlen(name_buf) != name_len)
    {
        addError(SOP_MESSAGE,
                 "Output attribute name is empty or contains a null byte.");
        return false;
    }

    const long tuple_size = PY_PyLong_AsLong(size_obj);
    if (tuple_size == -1 && PY_PyErr_Occurred())
    {
        reportPythonError("Bad tuple size in the Warp Code result");
        return false;
    }
    if (tuple_size != 1 && tuple_size != 3)
    {
        UT_WorkBuffer msg;
        msg.sprintf("Attribute '%s' has unsupported tuple size %d; only 1 "
                    "(float) and 3 (vector) are supported.",
                    name_buf, (int)tuple_size);
        addError(SOP_MESSAGE, msg.buffer());
        return false;
    }
    const int tsize = (int)tuple_size;

    char *data = nullptr;
    PY_Py_ssize_t nbytes = 0;
    if (PY_PyBytes_AsStringAndSize(data_obj, &data, &nbytes) != 0)
    {
        reportPythonError("Attribute data in the Warp Code result is not bytes");
        return false;
    }

    const PY_Py_ssize_t expected = (PY_Py_ssize_t)npoints
                                 * (PY_Py_ssize_t)tsize
                                 * (PY_Py_ssize_t)sizeof(fpreal32);
    if (nbytes != expected)
    {
        UT_WorkBuffer msg;
        msg.sprintf("Attribute '%s' carries %lld bytes but %lld are needed "
                    "for %lld points x %d float(s).",
                    name_buf, (long long)nbytes, (long long)expected,
                    (long long)npoints, tsize);
        addError(SOP_MESSAGE, msg.buffer());
        return false;
    }

    // Exact min AND max size: findFloatTuple()'s max_size defaults to -1,
    // meaning "no upper bound", so asking for 1 float would happily match a
    // 3-float attribute like P and then be written as if it were scalar.
    GA_Attribute *attrib =
        gdp->findFloatTuple(GA_ATTRIB_POINT, name_buf, tsize, tsize);
    if (!attrib)
    {
        if (gdp->findPointAttribute(name_buf) != nullptr)
        {
            // Deliberately not destroy-and-recreate: a stray
            // out = {"P": scalars} must fail loudly, not delete positions.
            UT_WorkBuffer msg;
            msg.sprintf("Point attribute '%s' already exists with an "
                        "incompatible type; cannot write %d float(s) to it.",
                        name_buf, tsize);
            addError(SOP_MESSAGE, msg.buffer());
            return false;
        }

        attrib = gdp->addFloatTuple(GA_ATTRIB_POINT, name_buf, tsize);
    }
    if (!attrib)
    {
        UT_WorkBuffer msg;
        msg.sprintf("Could not create point attribute '%s'.", name_buf);
        addError(SOP_MESSAGE, msg.buffer());
        return false;
    }

    // Row i belongs to point *number* i, which is how the HOM arrays on the
    // read side are indexed. GA_FOR_ALL_PTOFF walks point offsets, and offset
    // order only coincides with index order on a compacted detail, so map
    // through pointOffset() instead of assuming they match.
    //
    // Values are memcpy'd rather than read through a reinterpreted float
    // pointer: the byte buffer belongs to Python and nothing guarantees its
    // alignment or lets us alias it as fpreal32.
    if (tsize == 1)
    {
        GA_RWHandleF handle(attrib);
        if (!handle.isValid())
        {
            UT_WorkBuffer msg;
            msg.sprintf("Point attribute '%s' is not a writable float.",
                        name_buf);
            addError(SOP_MESSAGE, msg.buffer());
            return false;
        }

        for (GA_Size i = 0; i < npoints; ++i)
        {
            fpreal32 value;
            memcpy(&value, data + (size_t)i * sizeof(fpreal32),
                   sizeof(fpreal32));
            handle.set(gdp->pointOffset(GA_Index(i)), value);
        }
        handle.bumpDataId();
    }
    else
    {
        GA_RWHandleV3 handle(attrib);
        if (!handle.isValid())
        {
            UT_WorkBuffer msg;
            msg.sprintf("Point attribute '%s' is not a writable vector.",
                        name_buf);
            addError(SOP_MESSAGE, msg.buffer());
            return false;
        }

        for (GA_Size i = 0; i < npoints; ++i)
        {
            fpreal32 value[3];
            memcpy(value, data + (size_t)i * 3 * sizeof(fpreal32),
                   3 * sizeof(fpreal32));
            handle.set(gdp->pointOffset(GA_Index(i)),
                       UT_Vector3F(value[0], value[1], value[2]));
        }
        handle.bumpDataId();
    }

    return true;
}


bool
SOP_NvidiaWarp::evalRequiredString(const char *parm, fpreal t,
                                   UT_String &value, EmptyParm kind,
                                   const char *message)
{
    evalString(value, parm, 0, t);
    if (value.isstring())
        return true;

    if (kind == PASS_THROUGH)
        addWarning(SOP_MESSAGE, message);
    else
        addError(SOP_MESSAGE, message);
    return false;
}


OP_ERROR
SOP_NvidiaWarp::cookMySop(OP_Context &context)
{
    OP_AutoLockInputs inputs(this);
    if (inputs.lock(context) >= UT_ERROR_ABORT)
        return error();

    // gdp is populated by this guard, not here: see SourceGuard. Every exit
    // below is covered by its destructor, so no path can return empty
    // geometry by omission.
    SourceGuard source(*this, context);

    const fpreal t = context.getTime();

    if (!evalInt("enable", 0, t))
        return error();   // pass the input through untouched

    UT_String kernel_src;
    UT_String kernel_func;
    UT_String warp_code;
    UT_String out_attrib;

    // An empty code field passes geometry through, which is a warning. An
    // empty name is a misconfiguration, which is an error.
    if (!evalRequiredString("kernel", t, kernel_src, PASS_THROUGH,
                            "No kernel source; geometry passed through."))
        return error();
    if (!evalRequiredString("kernelname", t, kernel_func, MISCONFIGURED,
                            "Kernel Name is empty."))
        return error();
    if (!evalRequiredString("warpcode", t, warp_code, PASS_THROUGH,
                            "No Warp Code; geometry passed through."))
        return error();
    if (!evalRequiredString("outattrib", t, out_attrib, MISCONFIGURED,
                            "Output Attribute is empty."))
        return error();

    UT_String device;
    evalString(device, "device", 0, t);
    if (!device.isstring())
        device = "cpu";

    // The Warp Code field resolves its input geometry from this path. Passing
    // it explicitly beats hou.pwd(): it does not depend on HOM's "current
    // node" bookkeeping and it is trivially stubbable in tests.
    UT_String node_path;
    getFullPath(node_path);

    // Read from the input, not gdp: gdp is not populated yet.
    const GU_Detail *input_gdp = inputGeo(0);
    const GA_Size npoints = input_gdp ? input_gdp->getNumPoints() : 0;
    if (npoints == 0)
        return error();   // nothing to launch over

    // -----------------------------------------------------------------
    // Everything below touches Python. Houdini already owns the
    // interpreter, so we take the GIL with PY_InterpreterAutoLock and
    // never call Py_Initialize(). py_lock is constructed first, so it is
    // destroyed last: every PY_AutoObject below releases its reference
    // while the GIL is still held, on all paths including early returns.
    // -----------------------------------------------------------------
    {
        PY_InterpreterAutoLock py_lock;

        PY_AutoObject module(PY_PyImport_ImportModule(theWarpModule));
        if (!module.ptr())
        {
            reportPythonError("Could not import plattipus_nvidia_warp.kernel_utils");
            return error();
        }

        PY_AutoObject entry(
            PY_PyObject_GetAttrString(module, theWarpEntryPoint));
        if (!entry.ptr())
        {
            reportPythonError("kernel_utils has no run_warp_code()");
            return error();
        }

        // 'L' is long long: a detail with more than 2^31 points must not
        // silently wrap on the way into Python.
        PY_AutoObject args(PY_Py_BuildValue(
            "(ssssLss)",
            kernel_src.c_str(),
            kernel_func.c_str(),
            warp_code.c_str(),
            device.c_str(),
            (long long)npoints,
            out_attrib.c_str(),
            node_path.c_str()));
        if (!args.ptr())
        {
            reportPythonError("Could not build run_warp_code() arguments");
            return error();
        }

        PY_AutoObject result(PY_PyObject_CallObject(entry, args));
        if (!result.ptr())
        {
            // Covers a syntax error or a runtime failure in either code
            // field, a missing kernel name, a missing attribute, and a
            // malformed `out`. kernel_utils turns all of them into a
            // ValueError whose message lands on the node.
            reportPythonError("Warp Code failed");
            return error();
        }

        // run_warp_code() returns (results, adoption). `adoption` is None
        // unless the Warp Code touched houdiniGeo.
        if (!PY_PySequence_Check(result) || PY_PySequence_Size(result) != 2)
        {
            // PySequence_Size() may have set an error indicator reaching a
            // non-sequence; clear it so it cannot surface on a later call.
            PY_PyErr_Clear();
            addError(SOP_MESSAGE,
                     "run_warp_code() must return (results, adoption).");
            return error();
        }

        PY_AutoObject results(PY_PySequence_GetItem(result, 0));
        PY_AutoObject adoption(PY_PySequence_GetItem(result, 1));
        if (!results.ptr() || !adoption.ptr())
        {
            reportPythonError("Malformed run_warp_code() return value");
            return error();
        }

        // ------------------------------------------------------------------
        // houdiniGeo adoption.
        //
        // `adoption` owns the writable hou.Geometry AND its locked
        // GU_DetailHandle. The GU_Detail pointer below is valid only while
        // that Python object is alive. `adoption` is a PY_AutoObject living
        // to the end of this scope, so it outlives replaceWith().
        // Dropping it earlier would dangle the pointer and crash Houdini.
        // This mirrors what inlinecpp does: hold the handle for the duration
        // of the call, release it after.
        // ------------------------------------------------------------------
        GA_Size write_npoints = npoints;

        if (adoption.ptr() != PY_Py_None())
        {
            PY_AutoObject address(
                PY_PyObject_GetAttrString(adoption, "address"));
            if (!address.ptr())
            {
                reportPythonError("houdiniGeo adoption has no address");
                return error();
            }

            void *raw = PY_PyLong_AsVoidPtr(address);
            if (!raw || PY_PyErr_Occurred())
            {
                reportPythonError("houdiniGeo address is not a valid pointer");
                return error();
            }

            const GU_Detail *src = static_cast<const GU_Detail *>(raw);

            // houdiniGeo is frozen WITHOUT cloning data ids, so every
            // attribute on it carries a fresh id and replaceWith() copies all
            // of them. Cloning the ids would make replaceWith() skip edits to
            // attributes that already existed on the input, without
            // reporting an error.
            gdp->replaceWith(*src);

            // replaceWith() populated gdp; duplicating now would throw the
            // adopted geometry away.
            source.disarm();

            // Topology may have changed, so `out` arrays are sized against
            // the adopted geometry, not the input.
            write_npoints = gdp->getNumPoints();
        }
        else
        {
            // No adoption: gdp needs the input copied into it before any
            // `out` attribute is written onto it.
            source.duplicateNow();
        }

        // An empty result is legal: it is what `out = {}` means, and it is
        // also what an adopted houdiniGeo with no `out` produces.
        const PY_Py_ssize_t nresults = PY_PySequence_Size(results);
        for (PY_Py_ssize_t i = 0; i < nresults; ++i)
        {
            PY_AutoObject item(PY_PySequence_GetItem(results, i));
            if (!item.ptr())
            {
                reportPythonError("Bad item in the Warp Code result");
                return error();
            }

            if (!writeResultItem(item, write_npoints))
                return error();
        }
    }

    return error();
}


void
newSopOperator(OP_OperatorTable *table)
{
    OP_Operator *op = new OP_Operator(
        "plattipus::nvidia_warp::1.0",   // internal name
        "NVIDIA Warp",                   // UI label
        SOP_NvidiaWarp::myConstructor,
        SOP_NvidiaWarp::myTemplateList,
        1,   // min inputs
        1,   // max inputs
        0);  // no local variables

    // Resolves to config/Icons/plattipus_nvidia_warp.svg, which package.py puts on
    // HOUDINI_PATH. Houdini falls back to a default icon if not found, so a
    // missing file degrades quietly rather than failing to register.
    op->setIconName("plattipus_nvidia_warp");
    table->addOperator(op);
}
