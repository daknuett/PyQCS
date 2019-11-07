#include <Python.h>
#include <structmember.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>
#include <structmember.h>
#include <stdlib.h>


#include "vops.h"
#include "linked_list.h"
#include "graph_operations.h"


#include "../../gates/implementations/basic_gates.h"


static int
RawGraphState_init(RawGraphState * self
            , PyObject * args
            , PyObject * kwds)
{
    // FIXME: 
    // Do I actually want a new zero state here?
    static char * kwrds[] = {"length", NULL};
    npy_intp length;
    npy_intp i;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "I", kwrds, &length))
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
        self->vops[i] = VOP_H;
    }
    return 0;
}

/*
static PyObject *
RawGraphState_getitem(RawGraphState * self
                , PyObject * args)
{
    // Somehow PyArg_ParseTuple does utter crap if I 
    // do not set i and j explicitly before calling PyArg_ParseTuple.
    npy_intp i = 0xdeadbeef, j = 0xdeadbeef, swp, result;
    PyObject * result_obj;

    if(!PyArg_ParseTuple(args, "II", &i, &j))
    {
        return NULL;
    }

    if(i < j)
    {
        swp = i;
        i = j;
        j = swp;
    }

    if(j < 0)
    {
        PyErr_SetString(PyExc_KeyError, "index must be positive");
        return NULL;
    }
    if(i >= self->length)
    {
        PyErr_SetString(PyExc_KeyError, "index out of range");
        return NULL;
    }

    result  = ll_has_value(self->lists[i], j);
    result_obj = Py_BuildValue("i", result);
    return result_obj;
}

static PyObject *
RawGraphState_setitem(RawGraphState * self
                , PyObject * args)
{
    npy_intp i = 0xdeadbeef, j = 0xdeadbeef, swp, value = 0xdeadbeef;
    //npy_intp i, j, swp, value;
    int result;

    if(!PyArg_ParseTuple(args, "IIp", &i, &j, &value))
    {
        return NULL;
    }


    if(i < j)
    {
        swp = i;
        i = j;
        j = swp;
    }

    if(i >= self->length)
    {
        PyErr_SetString(PyExc_KeyError, "index out of range");
        return NULL;
    }

    if(value)
    {
        result = ll_insert_value(&(self->lists[i]), j);
        if(result == 2)
        {
            PyErr_SetString(PyExc_ValueError, "element is already set");
            return NULL;
        }
        if(result == 1)
        {
            PyErr_SetString(PyExc_MemoryError, "failed to allocate new node");
            return NULL;
        }
        result = ll_insert_value(&(self->lists[j]), i);
        if(result == 2)
        {
            PyErr_SetString(PyExc_ValueError, "element is already set");
            return NULL;
        }
        if(result == 1)
        {
            PyErr_SetString(PyExc_MemoryError, "failed to allocate new node");
            return NULL;
        }
        Py_RETURN_NONE;
    }
    else
    {
        result = ll_delete_value(&(self->lists[i]), j);

        if(result)
        {
            PyErr_SetString(PyExc_ValueError, "element is not set");
            return NULL;
        }
        result = ll_delete_value(&(self->lists[j]), i);

        if(result)
        {
            PyErr_SetString(PyExc_ValueError, "element is not set");
            return NULL;
        }
        Py_RETURN_NONE;
    }
}
*/

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

    self->vops[i] = vop_lookup_table[vop][self->vops[i]];

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
            if(PyList_Append(this_edges, PyLong_FromLong(1)) < 0)
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
RawGraphState_apply_CZ(RawGraphState * self, PyObject * args)
{
    npy_intp result;
    npy_intp i = 0, j = 0;
    if(!PyArg_ParseTuple(args, "II", &i, &j))
    {
        return NULL;
    }

    if(vop_commutes_with_CZ(self->vops[i]) && vop_commutes_with_CZ(self->vops[j]))
    {
        // Case 1
        result = graph_toggle_edge(self, i, j);
    }
    else
    {
        // Case 2
        if(ll_length(self->lists[i]) > 1 
            && ll_length(self->lists[j]) > 1)
        {
            //// Sub-Case 2.1
            //ll_node_t * ngbhd_i = self->lists[i];
            //ll_node_t * ngbhd_j = self->lists[j];

            //if(!vop_commutes_with_CZ(self->vops[i]))
            //{
            //    graph_La_transform_to_CZ_commuting(self, i
            //}

            PyErr_SetString(PyExc_NotImplementedError, "Work in progress");
            return NULL;

        }
        else
        {
            PyErr_SetString(PyExc_NotImplementedError, "To be done");
            return NULL;
        }
    }

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


static void
RawGraphState_dealloc(RawGraphState * self)
{
    int i;
    for(i = 0; i < self->length; i++)
    {
        ll_recursively_delete_list(self->lists[i]);
    }
    free(self->lists);
    free(self->vops);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyMemberDef RawGraphState_members[] = {{NULL}};
static PyMethodDef RawGraphState_methods[] = {
    //{"setitem", (PyCFunction) RawGraphState_setitem, METH_VARARGS, "sets an item"}
    //, {"getitem", (PyCFunction) RawGraphState_getitem, METH_VARARGS, "gets an item"}
    {"apply_C_L", (PyCFunction) RawGraphState_apply_C_L, METH_VARARGS, "applies a C_L operator"}
    , {"apply_CZ", (PyCFunction) RawGraphState_apply_CZ, METH_VARARGS, "applies a CZ operator"}
    , {"to_lists", (PyCFunction) RawGraphState_to_lists, METH_NOARGS, "converts the graph state to a python representation using lists"}
    , {NULL}
};

static PyTypeObject RawGraphStateType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pyqcs.graph.backed.raw_state.RawGraphState",
    .tp_doc = "special type for graph representation",
    .tp_basicsize = sizeof(RawGraphState),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc) RawGraphState_init,
    .tp_dealloc = (destructor) RawGraphState_dealloc,
    .tp_members = RawGraphState_members,
    .tp_methods = RawGraphState_methods,
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
    if(import_basic_gates() < 0)
    {
        return NULL;
    }
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
