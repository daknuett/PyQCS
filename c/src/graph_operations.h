#ifndef graph_operations_h_
#define graph_operations_h_

#include <inttypes.h>

#include "linked_list.h"

#define GRAPH_CLEAR_VOP_CANNOT_CLEAR_SECOND_VOP -4

typedef struct 
{
    long int length;
    ll_node_t ** lists;
    uint8_t * vops;
    
} RawGraphState;

// Used in graph_toggle_edge.
int
graph_toggle_edge_from_to(RawGraphState * self, long int i, long int j);

// Toggles the edge between i and j.
int
graph_toggle_edge(RawGraphState * self, long int i, long int j);

// Used in graph_clear_vop.
int
graph_La_transform(RawGraphState * self, long int i);

int
graph_isolated_two_qbit_CZ(RawGraphState * self, long int i, long int j);

int
graph_qbits_are_isolated(RawGraphState * self, long int i, long int j);

// Check whether the vop on i can be cleared while ignoring j.
int 
graph_can_clear_vop(RawGraphState * self, long int i, long int j);

/*
 * Clear the vop on a, ignoring the vertex b as a partner for graph_La_transform.
 * XXX: Note that this will result in a SIGSEGV if one does not check whether
 * the vop on a can be cleared while ignoring b.
 *
 * To check whether graph_clear_vop can be applied use graph_can_clear_vop.
 * */
int
graph_clear_vop(RawGraphState * self, long int a, long int b);

int
graph_update_after_measurement(RawGraphState * self
                            , uint8_t observable
                            , long int qbit
                            , long int result);
#endif
