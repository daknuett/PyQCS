#include <Python.h>
#include <numpy/ndarraytypes.h>
//#include <numpy/ufuncobject.h>
#include <structmember.h>
#include <stdlib.h>
#include "vops.h"

typedef struct ll_node_s
{
    struct ll_node_s * next;
    npy_intp value;
} ll_node_t;

void
ll_recursively_delete_list(ll_node_t * list)
{
    ll_node_t * next_node;
    while(list)
    {
        next_node = list->next;
        free(list);
        list = next_node;
    }
}

ll_node_t *
ll_node_t_new(ll_node_t * next, npy_intp value)
{
    ll_node_t * node = malloc(sizeof(ll_node_t));
    if(!node)
    {
        return NULL;
    }
    node->next = next;
    node->value = value;
    return node;
}

int
ll_insert_value(ll_node_t ** list, npy_intp value)
{
    ll_node_t * current_node;
    ll_node_t * last_node;
    ll_node_t * new_node;

    if(!*list)
    {
        *list = ll_node_t_new(NULL, value);
        if(*list)
        {
            return 0;
        }
        return 1;
    }

    current_node = *list;
    last_node = *list;
    while(current_node && current_node->value < value)
    {
        last_node = current_node;
        current_node = current_node->next;
        
    }

    if(current_node && current_node->value == value)
    {
        return 2;
    }

    
    new_node = ll_node_t_new(current_node, value);
    if(!new_node)
    {
        return 1;
    }
    // This is the case, when we set the first element.
    if(current_node == last_node)
    {   
        *list = new_node;
        return 0;
    }
    last_node->next = new_node;
    return 0;
}

int
ll_delete_value(ll_node_t ** list, npy_intp value)
{
    ll_node_t * current_node;
    ll_node_t * last_node;

    current_node = *list;
    last_node = *list;

    while(current_node && current_node->value < value)
    {
        last_node = current_node;
        current_node = current_node->next;
    }

    if(!current_node || current_node->value != value)
    {
        return 2;
    }

    if(current_node == last_node)
    {
        *list = current_node->next;
    }

    last_node->next = current_node->next;
    free(current_node);
    return 0;
}

int
ll_has_value(ll_node_t * list, npy_intp value)
{
    while(list && list->value < value)
    {
        list = list->next;
    }

    if(list && list->value == value)
    {
        return 1;
    }
    return 0;
}


typedef struct 
{
    PyObject_HEAD
    npy_intp length;
    ll_node_t ** lists;
    npy_uint8 * vops;
    
} RawGraphState;

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
        free(self->lists)
        PyErr_SetString(PyExc_MemoryError, "out of memory");
        return -1;
    }
    for(i = 0; i < length; i++)
    {
        vops[i] = VOP_H;
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
	// Somehow uninitialized values can make bad stuff.
    // So initialize this value with the least harmful operator.	
    npy_uint8 vop = VOP_I;
    npy_intp i = 0;

    if(!PyArg_ParseTuple(args, "II", &vop, &i))
    {
        return NULL;
    }

    if(vop >= 24)
    {
        PyErr_SetString(PyExc_ValueError, "vop index must be in [0, 23]");
        return NULL;
    }

    self->vops[i] = vop_lookup_table[vop][self->vops[i]];

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
#endif
