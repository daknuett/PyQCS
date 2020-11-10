#include <Python.h>
#include <structmember.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>
#include <structmember.h>
#include <stdlib.h>


#include "vops.h"
#include "linked_list.h"
#include "graph_operations.h"
#include "raw_state.h"

static PyTypeObject RawGraphStateType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pyqcs.graph.backed.raw_state.RawGraphState",
    .tp_doc = "special type for graph representation",
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyType_GenericNew,
};



static int
RawGraphState_init(RawGraphState * self
            , PyObject * args
            , PyObject * kwds)
{
    static char * kwrds[] = {"length", NULL};
    npy_intp length;
    npy_intp i;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "l", kwrds, &length))
    {
        return -1;
    }

    if(length <= 0)
    {
        PyErr_SetString(PyExc_ValueError, "length must be positive");
        return -1;
    }

    self->lists = calloc(sizeof(ll_node_t), length);
    self->vops = malloc(sizeof(npy_uint8) * length);
    self->length = length;
    if(!self->lists)
    {
        free(self->vops);
        PyErr_SetString(PyExc_MemoryError, "out of memory");
        return -1;
    }
    if(!self->vops)
    {
        free(self->lists);
        PyErr_SetString(PyExc_MemoryError, "out of memory");
        return -1;
    }
    for(i = 0; i < length; i++)
    {
        self->vops[i] = VOP_I;
    }
    return 0;
}

static PyObject *
RawGraphState_deepcopy(RawGraphState * self)
{
    PyObject * args;
    RawGraphState * new_graph;
    npy_intp i;

    args = Py_BuildValue("(I)", self->length);
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

    for(i = 0; i < self->length; i++)
    {
        new_graph->vops[i] = self->vops[i];
    }
    for(i = 0; i < self->length; i++)
    {
        if(ll_deepcopy(&(new_graph->lists[i]), &(self->lists[i])))
        {
            //ll_recursively_delete_list(new_graph->lists[i]);
            //new_graph->lists[i] = NULL;
            goto return_after_error;
        }
    }

    return (PyObject *) new_graph;

return_after_error:
    PyErr_SetString(PyExc_MemoryError, "failed to allocate new graph state: out of memory");
    Py_DECREF(new_graph);
    return NULL;

}


static PyObject *
RawGraphState_apply_C_L(RawGraphState * self
                        , PyObject * args)
{
    npy_uint8 vop;
    npy_intp i;

    if(!PyArg_ParseTuple(args, "lb", &i, &vop))
    {
        return NULL;
    }


    if(vop >= 24)
    {
        PyErr_SetString(PyExc_ValueError, "vop index must be in [0, 23]");
        return NULL;
    }

    if(i >= self->length)
    {
        PyErr_SetString(PyExc_ValueError, "qbit index out of range");
        return NULL;
    }

    graph_unchecked_apply_vop_left(self, i, vop);

    Py_RETURN_NONE;

}


static PyObject * 
RawGraphState_to_lists(RawGraphState * self)
{
    PyObject * vop_list;
    PyObject * adjacency_list;
    PyObject * this_edges;

    npy_intp i;

    vop_list = PyList_New(self->length);
    if(!vop_list)
    {
        return NULL;
    }
    adjacency_list = PyList_New(self->length);
    if(!adjacency_list)
    {
        Py_DECREF(vop_list);
        return NULL;
    }

    for(i = 0; i < self->length; i++)
    {
        PyList_SET_ITEM(vop_list, i, PyLong_FromLong(self->vops[i]));
        // XXX: This is crap. But it will do the trick for now.
        this_edges = PyList_New(0);
        ll_node_t * node = self->lists[i];
        while(node)
        {
            if(PyList_Append(this_edges, PyLong_FromLong(node->value)) < 0)
            {
                goto cleanup_error;
            }
            node = node->next;
        }
        PyList_SET_ITEM(adjacency_list, i, this_edges);
    }
    return PyTuple_Pack(2, vop_list, adjacency_list);

cleanup_error:
    Py_DECREF(this_edges);
    npy_intp j;
    for(j = 0; j < i; j++)
    {
        Py_DECREF(PyList_GET_ITEM(vop_list, j));
        Py_DECREF(PyList_GET_ITEM(adjacency_list, j));
    }
    Py_DECREF(PyList_GET_ITEM(vop_list, j));
    Py_DECREF(vop_list);
    Py_DECREF(adjacency_list);
    return NULL;
}

static PyObject *
RawGraphState_measure(RawGraphState * self, PyObject * args)
{
    npy_intp qbit;
    double random;
    npy_uint8 observable;
    npy_intp invert_result = 0;
    npy_intp result = 0;

    if(!PyArg_ParseTuple(args, "ld", &qbit, &random))
    {
        return NULL;
    }

    if(qbit > self->length)
    {
        PyErr_SetString(PyExc_ValueError, "qbit index out of range");
        return NULL;
    }

    observable = observable_after_vop_commute[0][self->vops[qbit]];
    if(observable > 2)
    {
        invert_result = 1;
    }

    // The only deterministic result, that also does not change
    // the graph state.
    if((observable == 2 || observable == 5) 
       && ll_length(self->lists[qbit]) == 0)
    {
       result = invert_result; // = 0 ^ invert_result
       return Py_BuildValue("l", result);
    }

    // Select the result randomly according to 
    // the given random number:
    if(random >= 0.5)
    {
        result = 1;
    }
    // invert_result means we are measuring -O instead of O.
    // This can be achieved by measuring O and inverting the result.
    // The state is changed according to the inverted result.
    if(invert_result)
    {
        result ^= 1;
        observable -= 3;
    }

    if(graph_update_after_measurement(self, observable, qbit, result))
    {
        return NULL;
    }
    // invert_result means we are measuring -O instead of O.
    // This can be achieved by measuring O and inverting the result.
    // The state is changed according to the inverted result.
    if(invert_result)
    {
        result ^= 1;
    }

    return Py_BuildValue("l", result);
}


static PyObject *
RawGraphState_apply_CZ(RawGraphState * self, PyObject * args)
{
    npy_intp result;
    npy_intp i = 0, j = 0;
    if(!PyArg_ParseTuple(args, "ll", &i, &j))
    {
        return NULL;
    }

    result = graph_do_apply_CZ(self, i, j);

    if(result == -2)
    {
        PyErr_SetString(PyExc_ValueError, "qbit indices out of rage");
        return NULL;
    }
    if(result < 0)
    {
        PyErr_SetString(PyExc_MemoryError, "failed to insert edge");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
RawGraphState_mul_to(RawGraphState * self, PyObject * args)
{
    RawGraphState * other;
    if(!PyArg_ParseTuple(args, "O!", &RawGraphStateType, &other))
    {
        return NULL;
    }

    if(self->length != other->length)
    {
        PyErr_SetString(PyExc_ValueError, "states must have same qbit count");
        return NULL;
    }

    // Multiply all VOPs to the right.
    npy_intp i, j;
    for(i = 0; i < self->length; i++)
    {
        graph_unchecked_apply_vop_left(self, i, daggered_vops[other->vops[i]]);
    }

    // Multiply all CZs to the right.
    for(i = 0; i < self->length; i++)
    {
        ll_iter_t * nbghd = ll_iter_t_new(other->lists[i]);
        while(ll_iter_next(nbghd, &j))
        {
            // Don't apply the same gate twice.
            if(j > i)
            {
                npy_intp result = graph_do_apply_CZ(self, i, j);

                if(result == -2)
                {
                    PyErr_SetString(PyExc_ValueError, "internal error: qbit index out of range");
                    return NULL;
                }
                if(result < 0)
                {
                    PyErr_SetString(PyExc_MemoryError, "failed to insert edge");
                    return NULL;
                }
            }
        }
        free(nbghd);
    }

    // The state on the left is now the trivial graph state, i.e. the <+|^n state.
    // The state we are operating on contains all the information.
    // We will now insert projection operators on the X +1 eigenstate. We can do so
    // because they act trivially on the state on the left. Because the projection
    // operator is hermitian we can also let it act on the ket on the right.



    npy_uint8 observable;
    double result = 1;
    npy_intp this_projection;
    npy_intp invert_result = 0;

    for(i = 0; i < self->length; i++)
    {
        this_projection = 0;
        observable = observable_after_vop_commute[2][self->vops[i]];
        if(observable > 2)
        {
            this_projection = 1;
        }

        // Projection on +/-X gives factor 1 or 0.
        // FIXME: use ll_is_empty here.
        if((observable == 2 || observable == 5)
           && ll_length(self->lists[i]) == 0)
        {
            if(this_projection)
            {
                return Py_BuildValue("l", 0);
            }
            else
            {
                if(graph_update_after_measurement(self, observable - 3, i, this_projection))
                {
                    return NULL;
                }
                continue;
            }
        }

        if(this_projection)
        {
            observable -= 3;
        }
        if(graph_update_after_measurement(self, observable, i, this_projection))
        {
            return NULL;
        }
        result *= M_SQRT1_2;
    }


    return Py_BuildValue("d", result);
}

static PyObject *
RawGraphState_project_to(RawGraphState * self, PyObject * args)
{
    npy_intp qbit = 0, observable = 0;
    npy_intp this_projection = 0;
    if(!PyArg_ParseTuple(args, "ll", &qbit, &observable))
    {
        return NULL;
    }
    if(qbit > self->length)
    {
        PyErr_SetString(PyExc_ValueError, "qbit index out of range");
        return NULL;
    }
    if((observable < 0) || (observable > 5))
    {
        PyErr_SetString(PyExc_ValueError, "observable must be in 0, ..., 5 (Z, Y, X, -Z, -Y, -X)");
        return NULL;
    }

    observable = observable_after_vop_commute[observable][self->vops[qbit]];

    if(observable > 2)
    {
        this_projection = 1;
    }

    // Projection on +/-X gives factor 1 or 0.
    // FIXME: use ll_is_empty here.
    if((observable == 2 || observable == 5)
       && ll_length(self->lists[qbit]) == 0)
    {
        if(this_projection)
        {
            if(graph_update_after_measurement(self, observable, qbit, this_projection))
            {
                return NULL;
            }
            return Py_BuildValue("l", 0);
        }
        else
        {
            return Py_BuildValue("l", 1);
        }
    }

    if(this_projection)
    {
        observable -= 3;
    }
    if(graph_update_after_measurement(self, observable, qbit, this_projection))
    {
        return NULL;
    }
    return Py_BuildValue("d", M_SQRT1_2);
}

static void
RawGraphState_dealloc(RawGraphState * self)
{
    npy_intp i;
    for(i = 0; i < self->length; i++)
    {
        ll_recursively_delete_list(&self->lists[i]);
    }
    free(self->lists);
    free(self->vops);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyMemberDef RawGraphState_members[] = {{NULL}};
static PyMethodDef RawGraphState_methods[] = {
    {"apply_C_L", (PyCFunction) RawGraphState_apply_C_L, METH_VARARGS, "applies a C_L operator"}
    , {"apply_CZ", (PyCFunction) RawGraphState_apply_CZ, METH_VARARGS, "applies a CZ operator"}
    , {"measure", (PyCFunction) RawGraphState_measure, METH_VARARGS, "measures a qbit"}
    , {"to_lists", (PyCFunction) RawGraphState_to_lists, METH_NOARGS, "converts the graph state to a python representation using lists"}
    , {"deepcopy", (PyCFunction) RawGraphState_deepcopy, METH_NOARGS, "deepcopy the graph"}
    , {"mul_to", (PyCFunction) RawGraphState_mul_to, METH_VARARGS, "computes overlap with other graph state; modifies self"}
    , {"project_to", (PyCFunction) RawGraphState_project_to, METH_VARARGS
                    , "the projection operator to the qbit; first argument is the qbit, second argument is the pauli index in "
                        "(Z, Y, X, -Z, -Y, -X)"
                            }
    , {NULL}
};


static PyModuleDef raw_statemodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "raw_state",
    .m_doc = "module containing the graph state class",
    .m_size = -1,
};


PyMODINIT_FUNC
PyInit_raw_state(void)
{
    RawGraphStateType.tp_methods = RawGraphState_methods;
    RawGraphStateType.tp_init = (initproc) RawGraphState_init;
    RawGraphStateType.tp_dealloc = (destructor) RawGraphState_dealloc;
    RawGraphStateType.tp_members = RawGraphState_members;
    RawGraphStateType.tp_basicsize = sizeof(RawGraphState);
    //if(import_array() < 0)
    //{
    //    return NULL;
    //}
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
