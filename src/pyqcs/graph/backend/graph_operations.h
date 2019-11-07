#ifndef graph_operations_h_
#define graph_operations_h_
#include <Python.h>
#include <structmember.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>
#include <structmember.h>

#include "linked_list.h"

typedef struct 
{
    PyObject_HEAD
    npy_intp length;
    ll_node_t ** lists;
    npy_uint8 * vops;
    
} RawGraphState;

npy_intp
graph_toggle_edge_from_to(RawGraphState * self, npy_intp i, npy_intp j);

npy_intp
graph_toggle_edge(RawGraphState * self, npy_intp i, npy_intp j);
#endif
