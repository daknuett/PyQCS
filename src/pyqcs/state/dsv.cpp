#include <Python.h>
#include <structmember.h>

#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>

#include <dsv.hpp>

#include <map>
#include <random>


typedef struct
{
    PyObject_HEAD
    dsv::DSV * state;
} RawDSVState;

static PyTypeObject RawDSVStateType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pyqcs.state.dsv.RawDSVState",
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "type for dsv representation",
    .tp_new = PyType_GenericNew,
};

static int RawDSVState_init(RawDSVState * self
            , PyObject * args
            , PyObject * kwds)
{
    static char * kwrds[] = {(char *)"nqbits", NULL};
    int nqbits;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "i", kwrds, &nqbits))
    {
        return -1;
    }

    self->state = new dsv::DSV(nqbits);

    return 0;
}

static PyObject * RawDSVState_deepcopy(RawDSVState * self)
{

    PyObject * args;
    RawDSVState * new_dsv;

    args = Py_BuildValue("(I)", self->state->nqbits());
    if(!args)
    {
        return NULL;
    }

    // CALL PYTHON CODE
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    new_dsv = (RawDSVState *) PyObject_CallObject((PyObject *) &RawDSVStateType, args);

    PyGILState_Release(gstate);
    // END CALL PYTHON CODE
    if(!new_dsv)
    {
        return NULL;
    }

    delete new_dsv->state;
    new_dsv->state = new dsv::DSV(*(self->state));

    return (PyObject *) new_dsv;
}

static void RawDSVState_dealloc(RawDSVState * self)
{
    delete self->state;
    Py_TYPE(self)->tp_free((PyObject *) self);
}


static PyObject * RawDSVState_apply_simple_gate(RawDSVState * self, PyObject * args)
{
    static std::map<std::string, dsv::dsv_op> simple_gates( {
        {"H", dsv::ops::H}, {"X", dsv::ops::X}, {"Z", dsv::ops::Z},
        {"S", dsv::ops::S}
    });

    int i = 0;
    char * gate_name = nullptr;
    if(!PyArg_ParseTuple(args, "is", &i, &gate_name))
    {
        return NULL;
    }

    if(i >= self->state->nqbits())
    {
        PyErr_SetString(PyExc_ValueError, "qbit out of range");
        return NULL;
    }

    auto find = simple_gates.find(gate_name);
    if(find == simple_gates.end())
    {
        PyErr_SetString(PyExc_ValueError, "unknown gate name (wrong application method?)");
        return NULL;

    }
    dsv::dsv_op op = *find->second;
    dsv::DSVOpArgument arg(i);

    self->state->apply_op(op, arg);
    Py_RETURN_NONE;
}
static PyObject * RawDSVState_apply_two_qbit_gate(RawDSVState * self, PyObject * args)
{
    static std::map<std::string, dsv::dsv_op> two_qbit_gates( {
            {"CZ", dsv::ops::CZ}, {"CX", dsv::ops::CX}
    });

    int i = 0, j = 0;
    char * gate_name = nullptr;
    if(!PyArg_ParseTuple(args, "iis", &i, &j, &gate_name))
    {
        return NULL;
    }

    if(i >= self->state->nqbits() || j >= self->state->nqbits())
    {
        PyErr_SetString(PyExc_ValueError, "qbit out of range");
        return NULL;
    }

    auto find = two_qbit_gates.find(gate_name);
    if(find == two_qbit_gates.end())
    {
        PyErr_SetString(PyExc_ValueError, "unknown gate name (wrong application method?)");
        return NULL;

    }
    dsv::dsv_op op = *find->second;
    dsv::DSVOpArgument arg(i, (short int) j);

    self->state->apply_op(op, arg);
    Py_RETURN_NONE;
}

static PyObject * RawDSVState_apply_parametric_gate(RawDSVState * self, PyObject * args)
{
    static std::map<std::string, dsv::dsv_op> parametric_gates( {
            {"R", dsv::ops::R}
    });

    int i = 0;
    double phi;
    char * gate_name = nullptr;
    if(!PyArg_ParseTuple(args, "ids", &i, &phi, &gate_name))
    {
        return NULL;
    }

    if(i >= self->state->nqbits())
    {
        PyErr_SetString(PyExc_ValueError, "qbit out of range");
        return NULL;
    }

    auto find = parametric_gates.find(gate_name);
    if(find == parametric_gates.end())
    {
        PyErr_SetString(PyExc_ValueError, "unknown gate name (wrong application method?)");
        return NULL;

    }
    dsv::dsv_op op = *find->second;
    dsv::DSVOpArgument arg(i, phi);

    self->state->apply_op(op, arg);
    Py_RETURN_NONE;
}

static PyObject * RawDSVState_randomize(RawDSVState * self, PyObject * args)
{
    int i = 0;
    if(!PyArg_ParseTuple(args, "i", &i))
    {
        return NULL;
    }
    std::mt19937_64 rne(i);
    self->state->randomize(rne);

    Py_RETURN_NONE;
}

static PyObject * RawDSVState_export_numpy(RawDSVState * self)
{
    size_t ndims;
    std::complex<double> * data;

    ndims = self->state->export_to_array(&data);

    npy_intp const dims[] = {(npy_intp) ndims, 0};
    PyObject * array = PyArray_SimpleNewFromData(1, dims, NPY_CDOUBLE, (void *) data);
    ((PyArrayObject *)array)->flags |= NPY_ARRAY_OWNDATA;

    return array;
}

static PyObject * RawDSVState_measure(RawDSVState * self, PyObject * args)
{
    int i = 0;
    double random = 0;
    if(!PyArg_ParseTuple(args, "id", &i, &random))
    {
        return NULL;
    }

    if(i >= self->state->nqbits())
    {
        PyErr_SetString(PyExc_ValueError, "qbit out of range");
        return NULL;
    }

    double probability = self->state->measurement_probability(i);

    if(probability > random)
    {
        self->state->project_to(i, 1);
        return Py_BuildValue("i", 1);
    }
    self->state->project_to(i, 0);
    return Py_BuildValue("i", 0);
}

static PyObject * RawDSVState_statistic(RawDSVState * self, PyObject * args)
{
    double eps = 0;
    if(!PyArg_ParseTuple(args, "d", &eps))
    {
        return NULL;
    }

    std::vector<unsigned int> labels;
    std::vector<double> probabilities;

    self->state->statistic(labels, probabilities, eps);

    npy_uint * labels_data = (npy_uint *) malloc(sizeof(*labels_data) * labels.size());
    if(!labels_data)
    {
        PyErr_SetString(PyExc_MemoryError, "failed to allocate memory for labels");
        return NULL;
    }
    double * probabilities_data = (double *) malloc(sizeof(*probabilities_data) * probabilities.size());
    if(!probabilities_data)
    {
        PyErr_SetString(PyExc_MemoryError, "failed to allocate memory for probabilities");
        free(labels_data);
        return NULL;
    }

    std::copy(labels.begin(), labels.end(), labels_data);
    std::copy(probabilities.begin(), probabilities.end(), probabilities_data);

    npy_intp const dims[] = {static_cast<npy_intp>(labels.size())};

    PyObject * labels_array = PyArray_SimpleNewFromData(1, dims, NPY_UINT, labels_data);
    PyObject * probabilities_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, probabilities_data);
    ((PyArrayObject *)labels_array)->flags |= NPY_ARRAY_OWNDATA;
    ((PyArrayObject *)probabilities_array)->flags |= NPY_ARRAY_OWNDATA;

    return PyTuple_Pack(2, labels_array, probabilities_array);
}

static PyObject * RawDSVState_overlap(RawDSVState * self, PyObject * args)
{
    RawDSVState * other;
    if(!PyArg_ParseTuple(args, "O!", &RawDSVStateType, &other))
    {
        return NULL;
    }

    if(self->state->nqbits() != other->state->nqbits())
    {
        PyErr_SetString(PyExc_ValueError, "states must have same qbit count");
        return NULL;
    }

    std::complex<double> result = (*self->state) * (*other->state);
    Py_complex * result_pc = (Py_complex *) calloc(1, sizeof(*result_pc));
    result_pc->real = result.real();
    result_pc->imag = result.imag();

    return Py_BuildValue("D", result_pc);

}

static PyObject * RawDSVState_redo_normalization(RawDSVState * self)
{
    self->state->normalize();
    Py_RETURN_NONE;
}

static PyObject * RawDSVState_project_Z(RawDSVState * self, PyObject * args)
{
    int i = 0, l = 0;
    double eps = 0;
    if(!PyArg_ParseTuple(args, "iid", &i, &l, &eps))
    {
        return NULL;
    }

    if(i >= self->state->nqbits())
    {
        PyErr_SetString(PyExc_ValueError, "qbit out of range");
        return NULL;
    }
    if(!(l == 1 || l == 0))
    {
        PyErr_SetString(PyExc_ValueError, "l must be 0 or 1");
        return NULL;
    }

    double amplitude_1 = self->state->measurement_probability(i);

    if(l == 0 && (1 - amplitude_1) < eps)
    {
        Py_RETURN_FALSE;
    }
    if(l == 1 && amplitude_1 < eps)
    {
        Py_RETURN_FALSE;
    }

    self->state->project_to(i, l);
    Py_RETURN_TRUE;
}

struct PyMemberDef RawDSVState_members[] = {{NULL}};
static PyMethodDef RawDSVState_methods[] = {
    {"deepcopy", (PyCFunction) RawDSVState_deepcopy, METH_NOARGS, "deepcopies the state"}
    , {"apply_simple_gate", (PyCFunction) RawDSVState_apply_simple_gate, METH_VARARGS, "applies a simple gate (takes only parameter act)"}
    , {"apply_two_qbit_gate", (PyCFunction) RawDSVState_apply_two_qbit_gate, METH_VARARGS, "applies a two qbit gate"}
    , {"apply_parametric_gate", (PyCFunction) RawDSVState_apply_parametric_gate, METH_VARARGS, "applies a parametric gate (takes a double parameter)"}
    , {"randomize", (PyCFunction)  RawDSVState_randomize, METH_VARARGS, "randomizes the state; takes an integer argument used as seed for a Mersenne-Twister RNG"}
    , {"export_numpy", (PyCFunction) RawDSVState_export_numpy, METH_NOARGS, "exports the state to a numpy array; the data is copied"}
    , {"measure", (PyCFunction) RawDSVState_measure, METH_VARARGS, "performs a measurement in computational basis; takes a random double [0, 1) and returns the result (0 or 1); collapses the state"}
    , {"statistic", (PyCFunction) RawDSVState_statistic, METH_VARARGS, "computes the probabilities for measure outcomes in computational bases and returns them in two numpy arrays (labels, probabilities); probabilities smaller than the double parameter are ignored"}
    , {"overlap", (PyCFunction) RawDSVState_overlap, METH_VARARGS, "computes the overlap between two RawDSVStates"}
    , {"redo_normalization", (PyCFunction) RawDSVState_redo_normalization, METH_NOARGS, "normalizes the state vector to 1"}
    , {"project_Z", (PyCFunction) RawDSVState_project_Z, METH_VARARGS, "project_Z(i, l, eps): project qbit i to value l, raise an error if the remaining amplitude is smaller than eps"}
    , {NULL}
};

static PyModuleDef dsv_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "dsv",
    .m_doc = "contains the dsv state class for python",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit_dsv(void)
{
    import_array();
    RawDSVStateType.tp_methods = RawDSVState_methods;
    RawDSVStateType.tp_init = (initproc) RawDSVState_init;
    RawDSVStateType.tp_dealloc = (destructor) RawDSVState_dealloc;
    RawDSVStateType.tp_members = RawDSVState_members;
    RawDSVStateType.tp_basicsize = sizeof(RawDSVState);
    PyObject * m;
    if(PyType_Ready(&RawDSVStateType) < 0)
    {
        return NULL;
    }

    m = PyModule_Create(&dsv_module);
    if(!m)
    {
        return NULL;
    }

    Py_INCREF(&RawDSVStateType);
    if(PyModule_AddObject(m, "RawDSVState", (PyObject *) &RawDSVStateType) < 0)
    {
        Py_DECREF(&RawDSVStateType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
