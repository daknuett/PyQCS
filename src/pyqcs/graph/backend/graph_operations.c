#include "graph_operations.h"
#include "vops.h"



int
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

int
graph_toggle_edge(RawGraphState * self, npy_intp i, npy_intp j)
{
    if(i < 0 || j < 0 || i >= self->length || j >= self->length || i == j)
    {
        return -2;
    }

    return (graph_toggle_edge_from_to(self, i, j)
            | graph_toggle_edge_from_to(self, j, i));

}

int
graph_clear_vops(RawGraphState * self, npy_intp a, npy_intp b)
{
    npy_uint8 product_length, vop;
    npy_intp i;

    // Start clearing with a:
    // Note that there is no need to clear the identity.
    if(self->vops[a] != VOP_I)
    {
        vop = self->vops[a];
        product_length = C_L_as_products_lengths[vop];

        for(i = product_length - 1; i >= 0; i++)
        {
            if(C_L_as_products[vop][i] == VOP_smiX)
            {
                graph_La_transform(self, a);
            }
            else
            {
                // self->lists[a]->value is guarranteed to exist
                // because we checked that both a and b have non-operand neighbours. 
                if(self->lists[a]->value != b)
                {
                    graph_La_transform(self, self->lists[a]->value);
                }
                else
                {
                    graph_La_transform(self, self->lists[a]->next->value);
                }
            }
        }

    }
    // Clear b:
    // Note that there is no need to clear the identity.
    if(self->vops[b] != VOP_I)
    {
        vop = self->vops[b];
        product_length = C_L_as_products_lengths[vop];

        for(i = product_length - 1; i >= 0; i++)
        {
            if(C_L_as_products[vop][i] == VOP_smiX)
            {
                graph_La_transform(self, b);
            }
            else
            {
                // self->lists[b]->value is guarranteed to exist
                // because we checked that both a and b have non-operand neighbours. 
                if(self->lists[b]->value != a)
                {
                    graph_La_transform(self, self->lists[b]->value);
                }
                else
                {
                    graph_La_transform(self, self->lists[b]->next->value);
                }
            }
        }
    }
    return 0;
}

int
graph_La_transform(RawGraphState * self, npy_intp i)
{
    ll_node_t * neighbours = self->lists[i];
    
    npy_intp b, c;

    ll_iter_t * iter_b = ll_iter_t_new(neighbours);
    ll_iter_t * iter_c = ll_iter_t_new(neighbours);

    self->vops[i] = vop_lookup_table[self->vops[i]][VOP_smiX];

    while(ll_iter_next(iter_b, &b))
    {
        self->vops[b] = vop_lookup_table[self->vops[b]][VOP_siZ];
        while(ll_iter_next(iter_c, &c))
        {
            // Do not re-toggle the edge.
            if(b == c)
            {
                break;
            }
            graph_toggle_edge(b, c);
        }
        ll_iter_reset(iter_c);
    }
    free(iter_b);
    free(iter_c);
    return 0;
}
