#ifndef graph_operations_h_
#define graph_operations_h_
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
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

// Used in graph_toggle_edge.
int
graph_toggle_edge_from_to(RawGraphState * self, npy_intp i, npy_intp j);

// Toggles the edge between i and j.
int
graph_toggle_edge(RawGraphState * self, npy_intp i, npy_intp j);

// Used in graph_clear_vops.
int
graph_La_transform(RawGraphState * self, npy_intp i);

// Used to clear the vops of two qbits if both qbits have non-operand
// neighbours.
int
graph_clear_vops(RawGraphState * self, npy_intp a, npy_intp b);

int
graph_isolated_two_qbit_CZ(RawGraphState * self, npy_intp i, npy_intp j);

int
graph_qbits_are_isolated(RawGraphState * self, npy_intp i, npy_intp j);

int
graph_clear_vop(RawGraphState * self, npy_intp a, npy_intp b);

int
graph_update_after_measurement(RawGraphState * self
                            , npy_uint8 observable
                            , npy_intp qbit
                            , npy_intp result);
#endif
