#include <Python.h>
#include <structmember.h>
#include <graphical.hpp>

typedef struct
{
    PyObject_HEAD
    graphical::GraphState * state;
} RawGraphState;

static PyTypeObject RawGraphStateType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pyqcs.graph.rawstate.RawGraphState",
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "special type for graph representation",
    .tp_new = PyType_GenericNew,
};

static int RawGraphState_init(RawGraphState * self
            , PyObject * args
            , PyObject * kwds)
{
    static char * kwrds[] = {(char *)"nqbits", NULL};
    int nqbits;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "i", kwrds, &nqbits))
    {
        return -1;
    }

    self->state = new graphical::GraphState(nqbits);

    return 0;
}

static PyObject * RawGraphState_deepcopy(RawGraphState * self)
{

    PyObject * args;
    RawGraphState * new_graph;

    args = Py_BuildValue("(I)", self->state->nqbits());
    if(!args)
    {
        return NULL;
    }

    // CALL PYTHON CODE
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    new_graph = (RawGraphState *) PyObject_CallObject((PyObject *) &RawGraphStateType, args);

    PyGILState_Release(gstate);
    // END CALL PYTHON CODE
    if(!new_graph)
    {
        return NULL;
    }

    delete new_graph->state;
    new_graph->state = new graphical::GraphState(*(self->state));

    return (PyObject *) new_graph;
}

static PyObject *
RawGraphState_apply_C_L(RawGraphState * self
                        , PyObject * args)
{
    uint8_t vop;
    int i;

    if(!PyArg_ParseTuple(args, "ib", &i, &vop))
    {
        return NULL;
    }


    if(vop >= 24)
    {
        PyErr_SetString(PyExc_ValueError, "vop index must be in [0, 23]");
        return NULL;
    }

    if(i >= self->state->nqbits())
    {
        PyErr_SetString(PyExc_ValueError, "qbit index out of range");
        return NULL;
    }

    self->state->apply_CL(i, vop);
    Py_RETURN_NONE;
}

static PyObject *
RawGraphState_apply_CZ(RawGraphState * self, PyObject * args)
{
    int i = 0, j = 0;
    if(!PyArg_ParseTuple(args, "ii", &i, &j))
    {
        return NULL;
    }


    self->state->apply_CZ(i, j);

    Py_RETURN_NONE;
}

static PyObject * 
RawGraphState_to_lists(RawGraphState * self)
{
    PyObject * vop_list;
    PyObject * edges_list;

    std::vector<int> vops;
    std::vector<std::vector<int>> edges;

    self->state->export_to_vectors(vops, edges);

    vop_list = PyList_New(vops.size());
    if(!vop_list)
    {
        return NULL;
    }
    edges_list = PyList_New(edges.size());
    if(!edges_list)
    {
        Py_DECREF(vop_list);
        return NULL;
    }

    for(size_t i = 0; i < vops.size(); i++)
    {
        PyObject * item = PyLong_FromLong(vops[i]);
        if(!item)
        {
            Py_DECREF(vop_list);
            Py_DECREF(edges_list);
            throw std::logic_error("Failed to allocate memory for Python list.");
        }

        PyList_SET_ITEM(vop_list, i, item);
    }

    for(size_t i = 0; i < edges.size(); i++)
    {
        PyObject * this_list = PyList_New(edges[i].size());
        if(!this_list)
        {
            Py_DECREF(vop_list);
            Py_DECREF(edges_list);
            throw std::logic_error("Failed to allocate memory for Python list.");
        }
        PyList_SET_ITEM(edges_list, i, this_list);

        for(size_t j = 0; j < edges[i].size(); j++)
        {
            PyObject * item = PyLong_FromLong(edges[i][j]);
            if(!item)
            {
                Py_DECREF(vop_list);
                Py_DECREF(edges_list);
                throw std::logic_error("Failed to allocate memory for Python list.");
            }
            PyList_SET_ITEM(this_list, j, item);
        }
    }

    return PyTuple_Pack(2, vop_list, edges_list);
}

static PyObject *
RawGraphState_measure(RawGraphState * self, PyObject * args)
{
    int qbit;
    double random;

    if(!PyArg_ParseTuple(args, "id", &qbit, &random))
    {
        return NULL;
    }

    if(qbit > self->state->nqbits())
    {
        PyErr_SetString(PyExc_ValueError, "qbit index out of range");
        return NULL;
    }

    int probability = self->state->measurement_probability(qbit, graphical::pauli_Z);

    if(probability == 0)
    {
        return Py_BuildValue("l", 0);
    }
    if(probability == -1)
    {
        return Py_BuildValue("l", 1);
    }

    // Probability is now 1 => indertiministic result.
    
    if(random >= 0.5)
    {
        self->state->project_to(qbit, graphical::pauli_mZ);
        return Py_BuildValue("l", 1);
    }
    self->state->project_to(qbit, graphical::pauli_Z);
    return Py_BuildValue("l", 0);
}


static void
RawGraphState_dealloc(RawGraphState * self)
{
    delete self->state;
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *
RawGraphState_project_to(RawGraphState * self, PyObject * args)
{
    int qbit = 0, observable = 0;
    if(!PyArg_ParseTuple(args, "ii", &qbit, &observable))
    {
        return NULL;
    }
    if(qbit > self->state->nqbits())
    {
        PyErr_SetString(PyExc_ValueError, "qbit index out of range");
        return NULL;
    }
    if((observable < 0) || (observable > 5))
    {
        PyErr_SetString(PyExc_ValueError, "observable must be in 0, ..., 5 (Z, Y, X, -Z, -Y, -X)");
        return NULL;
    }

    int projective_probability = self->state->measurement_probability(qbit, observable);
    if(projective_probability == -1)
    {
        return Py_BuildValue("l", 0);
    }
    if(projective_probability == 0)
    {
        return Py_BuildValue("l", 1);
    }

    self->state->project_to(qbit, observable);
    return Py_BuildValue("d", M_SQRT1_2);
}

static PyObject *
RawGraphState_mul_to(RawGraphState * self, PyObject * args)
{
    RawGraphState * other;
    if(!PyArg_ParseTuple(args, "O!", &RawGraphStateType, &other))
    {
        return NULL;
    }

    if(self->state->nqbits() != other->state->nqbits())
    {
        PyErr_SetString(PyExc_ValueError, "states must have same qbit count");
        return NULL;
    }

    int result = (*self->state) * (*other->state);

    if(result < 0)
    {
        return Py_BuildValue("i", 0);
    }
    if(result == 0)
    {
        return Py_BuildValue("i", 1);
    }
    return Py_BuildValue("d", std::pow(M_SQRT1_2, result));
}

static PyMemberDef RawGraphState_members[] = {{NULL}};
static PyMethodDef RawGraphState_methods[] = {
    {"apply_C_L", (PyCFunction) RawGraphState_apply_C_L, METH_VARARGS, "applies a C_L operator"}
    , {"apply_CZ", (PyCFunction) RawGraphState_apply_CZ, METH_VARARGS, "applies a CZ operator"}
    , {"measure", (PyCFunction) RawGraphState_measure, METH_VARARGS, "measures a qbit"}
    , {"to_lists", (PyCFunction) RawGraphState_to_lists, METH_NOARGS, "converts the graph state to a python representation using lists"}
    , {"deepcopy", (PyCFunction) RawGraphState_deepcopy, METH_NOARGS, "deepcopy the graph"}
    , {"mul_to", (PyCFunction) RawGraphState_mul_to, METH_VARARGS, "computes overlap with other graph state; internally copies self."}
    , {"project_to", (PyCFunction) RawGraphState_project_to, METH_VARARGS
                    , "the projection operator to the qbit; first argument is the qbit, second argument is the pauli index in "
                        "(Z, Y, X, -Z, -Y, -X)"
                            }
    , {NULL}
};

static PyModuleDef raw_statemodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "rawstate",
    .m_doc = "module containing the graph state class",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit_rawstate(void)
{
    RawGraphStateType.tp_methods = RawGraphState_methods;
    RawGraphStateType.tp_init = (initproc) RawGraphState_init;
    RawGraphStateType.tp_dealloc = (destructor) RawGraphState_dealloc;
    RawGraphStateType.tp_members = RawGraphState_members;
    RawGraphStateType.tp_basicsize = sizeof(RawGraphState);
    PyObject * m;
    if(PyType_Ready(&RawGraphStateType) < 0)
    {
        return NULL;
    }

    m = PyModule_Create(&raw_statemodule);
    if(!m)
    {
        return NULL;
    }

    Py_INCREF(&RawGraphStateType);
    if(PyModule_AddObject(m, "RawGraphState", (PyObject *) &RawGraphStateType) < 0)
    {
        Py_DECREF(&RawGraphStateType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
