#ifndef graph_operations_h_
#define graph_operations_h_
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <structmember.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>
#include <structmember.h>

#include "linked_list.h"

#define GRAPH_CLEAR_VOP_CANNOT_CLEAR_SECOND_VOP -4

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

// Used in graph_clear_vop.
int
graph_La_transform(RawGraphState * self, npy_intp i);

int
graph_isolated_two_qbit_CZ(RawGraphState * self, npy_intp i, npy_intp j);

int
graph_qbits_are_isolated(RawGraphState * self, npy_intp i, npy_intp j);

// Check whether the vop on i can be cleared while ignoring j.
int 
graph_can_clear_vop(RawGraphState * self, npy_intp i, npy_intp j);

/*
 * Clear the vop on a, ignoring the vertex b as a partner for graph_La_transform.
 * XXX: Note that this will result in a SIGSEGV if one does not check whether
 * the vop on a can be cleared while ignoring b.
 *
 * To check whether graph_clear_vop can be applied use graph_can_clear_vop.
 * */
int
graph_clear_vop(RawGraphState * self, npy_intp a, npy_intp b);

int
graph_update_after_measurement(RawGraphState * self
                            , npy_uint8 observable
                            , npy_intp qbit
                            , npy_intp result);
#endif
