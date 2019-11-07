#include "graph_operations.h"



npy_intp
graph_toggle_edge_from_to(RawGraphState * self, npy_intp i, npy_intp j)
{
    ll_node_t * list = self->lists[i];
    ll_node_t * last = self->lists[i];

    while(list && list->value < j)
    {
        last = list;
        list = list->next;
    }
    if(!list)
    {
        return ll_insert_value(&last, j);
    }

    if(list->value == j)
    {
        last->next = list->next;
        free(list);
        return 0;
    }
    ll_node_t * new_node = ll_node_t_new(list, j);
    last->next = new_node;
    return 0;
}

npy_intp
graph_toggle_edge(RawGraphState * self, npy_intp i, npy_intp j)
{
    if(i < 0 || j < 0 || i >= self->length || j >= self->length || i == j)
    {
        return -2;
    }

    return (graph_toggle_edge_from_to(self, i, j)
            | graph_toggle_edge_from_to(self, j, i));

}
