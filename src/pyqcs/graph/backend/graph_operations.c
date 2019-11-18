#include "graph_operations.h"
#include "vops.h"



int
graph_toggle_edge_from_to(RawGraphState * self, npy_intp i, npy_intp j)
{
#ifdef G_USE_OPTIMIZED_TOGGLE_EDGE_FROM_TO
#warning "G_USE_OPTIMIZED_TOGGLE_EDGE_FROM_TO is untested and might yield unexpected bugs"
#error "there is a known bug in G_USE_OPTIMIZED_TOGGLE_EDGE_FROM_TO"
    ll_node_t * list = self->lists[i];
    ll_node_t * last = self->lists[i];
    ll_node_t * new_node;
    printf("toggling edge %ld -> %ld\n", i, j);

    if(!list)
    {
        return ll_insert_value(&(self->lists[i]), j);
    }
    while(list && list->value < j)
    {
        printf("searching node (current: %ld) ... \n", list->value);
        last = list;
        list = list->next;
    }
    if(!list)
    {
        printf("appending node %ld\n", j);
        new_node = ll_node_t_new(NULL, j);
        if(!new_node)
        {
            return -1;
        }
        last->next = new_node;
        return 0;
    }

    if(list->value == j)
    {
        last->next = list->next;
        free(list);
        return 0;
    }
    if(last == list)
    {
        return ll_insert_value(&(self->lists[i]), j);
    }
    new_node = ll_node_t_new(list, j);
    if(!new_node)
    {
        return -1;
    }

    last->next = new_node;
    return 0;
#else
    if(ll_has_value(self->lists[i], j))
    {
        return ll_delete_value(&(self->lists[i]), j);
    }
    return ll_insert_value(&(self->lists[i]), j);

#endif
}

int
graph_toggle_edge(RawGraphState * self, npy_intp i, npy_intp j)
{
    int result;
    if(i < 0 || j < 0 || i >= self->length || j >= self->length || i == j)
    {
        return -2;
    }

    result = graph_toggle_edge_from_to(self, i, j);
    if(result != 0)
    {
        return result;
    }
    result = graph_toggle_edge_from_to(self, j, i);
    // XXX: no rollback can be done here. State is now corrupted!
    return result;
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

        for(i = product_length - 1; i >= 0; i--)
        {
            if(C_L_as_products[vop][i] == VOP_smiX)
            {
                graph_La_transform(self, a);
            }
            else
            {
                // self->lists[a]->value is guaranteed to exist
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

        for(i = product_length - 1; i >= 0; i--)
        {
            if(C_L_as_products[vop][i] == VOP_smiX)
            {
                graph_La_transform(self, b);
            }
            else
            {
                // self->lists[b]->value is guaranteed to exist
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
            graph_toggle_edge(self, b, c);
        }
        ll_iter_reset(iter_c);
    }
    free(iter_b);
    free(iter_c);
    return 0;
}

int
graph_isolated_two_qbit_CZ(RawGraphState * self, npy_intp i, npy_intp j)
{
    npy_intp is_CZ = ll_has_value(self->lists[i], j);
    npy_intp lookup_table_index = two_qbit_config_to_number[self->vops[i]][self->vops[j]][is_CZ];
    
    self->vops[i] = two_qbit_vops_after_CZ[lookup_table_index][0];
    self->vops[j] = two_qbit_vops_after_CZ[lookup_table_index][1];

    npy_intp is_entangled = two_qbit_vops_after_CZ[lookup_table_index][2];

    if(is_entangled)
    {
        if(is_CZ)
        {
            return 0;
        }
        return graph_toggle_edge(self, i, j);
    }
    if(is_CZ)
    {
        return graph_toggle_edge(self, i, j);
    }
    return 0;
}

int
graph_qbits_are_isolated(RawGraphState * self, npy_intp i, npy_intp j)
{
    npy_intp length_i = ll_length(self->lists[i]);
    npy_intp length_j = ll_length(self->lists[j]);
    if(length_i > 1 || length_j > 1)
    {
        return 0;
    }

    if(ll_has_value(self->lists[i], j))
    {
        return 1;
    }
    if(length_i == 0 && length_j == 0)
    {
        return 1;
    }
    return 0;
}

int
graph_clear_vop(RawGraphState * self, npy_intp a, npy_intp b)
{
    npy_uint8 product_length, vop;
    npy_intp i;

    // Start clearing with a:
    // Note that there is no need to clear the identity.
    if(self->vops[a] != VOP_I)
    {
        vop = self->vops[a];
        product_length = C_L_as_products_lengths[vop];

        for(i = product_length - 1; i >= 0; i--)
        {
            if(C_L_as_products[vop][i] == VOP_smiX)
            {
                graph_La_transform(self, a);
            }
            else
            {
                // self->lists[a]->value is guaranteed to exist
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
    return 0;
}