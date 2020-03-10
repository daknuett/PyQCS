#include <stdlib.h>

#include "vops.h"
#include "linked_list.h"
#include "graph_operations.h"
#include "raw_state.h"
#include "error.h"


int
RawGraphState_init(RawGraphState * self
            , long int length
            )
{
    char * kwrds[] = {"length", NULL};
    long int i;

    if(length <= 0)
    {
        return -1;
    }

    self->lists = calloc(sizeof(ll_node_t), length);
    self->vops = malloc(sizeof(uint8_t) * length);
    self->length = length;
    if(!self->lists)
    {
        GQCS_set_error("failed to allocate graph");
        free(self->vops);
        return -1;
    }
    if(!self->vops)
    {
        GQCS_set_error("failed to allocate graph");
        free(self->lists);
        return -1;
    }
    for(i = 0; i < length; i++)
    {
        self->vops[i] = VOP_I;
    }
    return 0;
}

RawGraphState *
RawGraphState_deepcopy(RawGraphState * self)
{
    RawGraphState * new_graph;
    long int i;

    new_graph = malloc(sizeof(RawGraphState));
    if(!new_graph)
    {
        GQCS_set_error("failed to allocate graph");
        return NULL;
    }

    if(RawGraphState_init(new_graph, self->length) < 0)
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

    return new_graph;

return_after_error:
    GQCS_set_error("failed to allocate graph");
    RawGraphState_dealloc(new_graph);
    return NULL;

}


int
RawGraphState_apply_C_L(RawGraphState * self
                        , long int i
                        , uint8_t vop)
{
    if(vop >= 24)
    {
        GQCS_set_error("vop index must be in [0, 23]");
        return -1;
    }

    if(i >= self->length)
    {
        GQCS_set_error("qbit index out of range");
        return -1;
    }

    self->vops[i] = vop_lookup_table[vop][self->vops[i]];

    return 0;

}

//PyObject * 
//RawGraphState_to_lists(RawGraphState * self)
//{
//    PyObject * vop_list;
//    PyObject * adjacency_list;
//    PyObject * this_edges;
//
//    long int i;
//
//    vop_list = PyList_New(self->length);
//    if(!vop_list)
//    {
//        return NULL;
//    }
//    adjacency_list = PyList_New(self->length);
//    if(!adjacency_list)
//    {
//        Py_DECREF(vop_list);
//        return NULL;
//    }
//
//    for(i = 0; i < self->length; i++)
//    {
//        PyList_SET_ITEM(vop_list, i, PyLong_FromLong(self->vops[i]));
//        // XXX: This is crap. But it will do the trick for now.
//        this_edges = PyList_New(0);
//        ll_node_t * node = self->lists[i];
//        while(node)
//        {
//            if(PyList_Append(this_edges, PyLong_FromLong(node->value)) < 0)
//            {
//                goto cleanup_error;
//            }
//            node = node->next;
//        }
//        PyList_SET_ITEM(adjacency_list, i, this_edges);
//    }
//    return PyTuple_Pack(2, vop_list, adjacency_list);
//
//cleanup_error:
//    Py_DECREF(this_edges);
//    long int j;
//    for(j = 0; j < i; j++)
//    {
//        Py_DECREF(PyList_GET_ITEM(vop_list, j));
//        Py_DECREF(PyList_GET_ITEM(adjacency_list, j));
//    }
//    Py_DECREF(PyList_GET_ITEM(vop_list, j));
//    Py_DECREF(vop_list);
//    Py_DECREF(adjacency_list);
//    return NULL;
//}

GQCS_measurement_result *
RawGraphState_measure(RawGraphState * self, long int qbit, double random)
{
    uint8_t observable;
    long int invert_result = 0;
    long int result = 0;

    if(qbit > self->length)
    {
        GQCS_set_error("qbit index out of range");
        return NULL;
    }

    observable = observable_after_vop_commute[self->vops[qbit]];
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
       
       GQCS_measurement_result * mres = malloc(sizeof(GQCS_measurement_result));
       if(!mres)
       {
           GQCS_set_error("failed to allocate result");
           return NULL;
       }
       mres->qbit = qbit;
       mres->result = result;
       return mres;
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

   GQCS_measurement_result * mres = malloc(sizeof(GQCS_measurement_result));
   if(!mres)
   {
       GQCS_set_error("failed to allocate result");
       return NULL;
   }
   mres->qbit = qbit;
   mres->result = result;
   return mres;
}


int
RawGraphState_apply_CZ(RawGraphState * self, long int i, long int j)
{
    long int result;

    if(vop_commutes_with_CZ(self->vops[i]) && vop_commutes_with_CZ(self->vops[j]))
    {
        // Case 1
        result = graph_toggle_edge(self, i, j);
        goto rs_CZ_exit;
    }
    // From now on Case 2.
    if(graph_qbits_are_isolated(self, i, j))
    {
        // Sub-Sub-Case 2.2.1
        result = graph_isolated_two_qbit_CZ(self, i, j);
        goto rs_CZ_exit;
    }
    int cleared_i = 0;
    int cleared_j = 0;
    if(graph_can_clear_vop(self, i, j))
    {
        cleared_i = 1;
        result = graph_clear_vop(self, i, j);
        if(result)
        {
            goto rs_CZ_exit;
        }
    }
    if(graph_can_clear_vop(self, j, i))
    {
        cleared_j = 1;
        result = graph_clear_vop(self, j, i);
        if(result)
        {
            goto rs_CZ_exit;
        }
    }
    if(!cleared_i && graph_can_clear_vop(self, i, j))
    {
        cleared_i = 1;
        result = graph_clear_vop(self, i, j);
        if(result)
        {
            goto rs_CZ_exit;
        }
    }

    if(cleared_i && cleared_j)
    {
        // Sub-Case 2.1
        result = graph_toggle_edge(self, i, j);
        goto rs_CZ_exit;
    }

    // Sub-Sub-Case 2.2.2
    result = graph_isolated_two_qbit_CZ(self, i, j);


rs_CZ_exit:

    if(result == -2)
    {
        GQCS_set_error("qbit indices out of rage");
        return -1;
    }
    if(result < 0)
    {
        GQCS_set_error("failed to insert edge");
        return -1;
    }
    return 0;
}


void
RawGraphState_dealloc(RawGraphState * self)
{
    int i;
    for(i = 0; i < self->length; i++)
    {
        ll_recursively_delete_list(&self->lists[i]);
    }
    free(self->lists);
    free(self->vops);
    free(self);
}
