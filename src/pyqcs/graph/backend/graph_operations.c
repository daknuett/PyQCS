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
graph_La_transform(RawGraphState * self, npy_intp i)
{
    int result = 0;
    ll_node_t * neighbours = self->lists[i];
    
    npy_intp b, c;

    // Probability for problems to occur here is small enough.
    // Don't do error checking for now. FIXME.
    ll_iter_t * iter_b = ll_iter_t_new(neighbours);
    ll_iter_t * iter_c = ll_iter_t_new(neighbours);

    self->vops[i] = vop_lookup_table[self->vops[i]][VOP_siX];


    while(ll_iter_next(iter_b, &b))
    {
        self->vops[b] = vop_lookup_table[self->vops[b]][VOP_smiZ];
        while(ll_iter_next(iter_c, &c))
        {
            // Do not re-toggle the edge.
            if(b == c)
            {
                break;
            }
            //if(c == i)
            //{
            //    continue;
            //}
            result = graph_toggle_edge(self, b, c);
            if(result)
            {
                goto lat_exit;
            }
        }
        ll_iter_reset(iter_c);
    }
lat_exit:
    free(iter_b);
    free(iter_c);
    return result;
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
    int result = 0;

    // Start clearing with a:
    // Note that there is no need to clear the identity.
    if(self->vops[a] != VOP_I)
    {
        vop = self->vops[a];
        product_length = C_L_as_products_lengths[vop];

        for(i = product_length - 1; i >= 0; i--)
        {
            if(C_L_as_products_daggered[vop][i] == VOP_siX)
            {
                result = graph_La_transform(self, a);
                if(result)
                {
                    goto gcv_exit;
                }
            }
            else
            {
                // self->lists[a]->value is guaranteed to exist
                // because we checked that both a and b have non-operand neighbours. 
                if(self->lists[a]->value != b)
                {
                    result = graph_La_transform(self, self->lists[a]->value);
                    if(result)
                    {
                        goto gcv_exit;
                    }
                }
                else
                {
                    result = graph_La_transform(self, self->lists[a]->next->value);
                    if(result)
                    {
                        goto gcv_exit;
                    }
                }
            }
        }
    }
gcv_exit:
    return result;
}

int
graph_isolate_qbit(RawGraphState * self
                , npy_intp qbit)
{
    ll_iter_t * iter = ll_iter_t_new(self->lists[qbit]);
    npy_intp ngb;
    while(ll_iter_next(iter, &ngb))
    {
        ll_delete_value(&self->lists[ngb], qbit);
    }
    free(iter);
    ll_recursively_delete_list(&self->lists[qbit]);
    return 0;
}

int
graph_toggle_neighbourhood(RawGraphState * self
                        , npy_intp qbit)
{
    ll_node_t * neighbours = self->lists[qbit];

    npy_intp b, c;

    ll_iter_t * iter_b = ll_iter_t_new(neighbours);
    ll_iter_t * iter_c = ll_iter_t_new(neighbours);

    while(ll_iter_next(iter_b, &b))
    {
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
graph_update_after_X_measurement(RawGraphState * self
                            , npy_intp a
                            , npy_intp result)
{
    // We are ensured that this list has at least one element, 
    // as the case of the isolated a is handled explicitly in RawGraphState_measure.
    npy_intp b = self->lists[a]->value;

    if(!result)
    {
        // Update the VOPs
        self->vops[a] = vop_lookup_table[self->vops[a]][projected_vop[2]];
        self->vops[b] = vop_lookup_table[self->vops[b]][VOP_smiY];

        ll_iter_t * iter_c = ll_iter_t_new(self->lists[a]);
        npy_intp c;
        while(ll_iter_next(iter_c, &c))
        {
            if(c != b && !ll_has_value(self->lists[b], c))
            {
                self->vops[c] = vop_lookup_table[self->vops[c]][VOP_Z];
            }
        }
        free(iter_c);
    }
    else
    {
        // Update the VOPs
        self->vops[a] = vop_lookup_table[self->vops[a]][projected_vop[5]];
        self->vops[b] = vop_lookup_table[self->vops[b]][VOP_siY];

        ll_iter_t * iter_c = ll_iter_t_new(self->lists[b]);
        npy_intp c;
        while(ll_iter_next(iter_c, &c))
        {
            if(c != a && !ll_has_value(self->lists[a], c))
            {
                self->vops[c] = vop_lookup_table[self->vops[c]][VOP_Z];
            }
        }
        free(iter_c);
    }

    ll_node_t * ngbh_a = NULL;
    ll_node_t * ngbh_b = NULL;
    if(ll_deepcopy(&ngbh_a, &self->lists[a]))
    {
        PyErr_SetString(PyExc_MemoryError, "failed to copy neighbourhood of qbit a");
        return -1;
    }
    if(ll_deepcopy(&ngbh_b, &self->lists[b]))
    {
        PyErr_SetString(PyExc_MemoryError, "failed to copy neighbourhood of qbit b");
        ll_recursively_delete_list(&ngbh_a);
        return -1;
    }
    ll_iter_t * iter_c = ll_iter_t_new(ngbh_b);
    ll_iter_t * iter_d = ll_iter_t_new(ngbh_a);
    npy_intp c, d;

    while(ll_iter_next(iter_c, &c))
    {
        ll_iter_reset(iter_d);
        while(ll_iter_next(iter_d, &d))
        {
            if(c == d)
            {
                continue;
            }
            graph_toggle_edge(self, c, d);
        }
    }

    ll_iter_reset(iter_c);
    free(iter_d);
    iter_d = ll_iter_t_new(ngbh_b);
    while(ll_iter_next(iter_c, &c))
    {
        if(!ll_has_value(self->lists[a], c))
        {
            continue;
        }
        ll_iter_reset(iter_d);
        while(ll_iter_next(iter_d, &d))
        {
            if(d == c)
            {
                continue;
            }
            if(!ll_has_value(self->lists[a], d))
            {
                continue;
            }
            graph_toggle_edge(self, c, d);
        }
    }

    free(iter_c);
    free(iter_d);

    iter_d = ll_iter_t_new(ngbh_a);
    while(ll_iter_next(iter_d, &d))
    {
        if(d == b)
        {
            continue;
        }
        graph_toggle_edge(self, d, b);
    }
    free(iter_d);
    ll_recursively_delete_list(&ngbh_a);
    ll_recursively_delete_list(&ngbh_b);
    return 0;
}

int
graph_update_after_Y_measurement(RawGraphState * self
                            , npy_intp qbit
                            , npy_intp result)
{
    if(!result)
    {
        self->vops[qbit] = vop_lookup_table[self->vops[qbit]][projected_vop[1]];
        ll_iter_t * iter = ll_iter_t_new(self->lists[qbit]);
        npy_intp neighbour;
        while(ll_iter_next(iter, &neighbour))
        {
            self->vops[neighbour] = vop_lookup_table[self->vops[neighbour]][VOP_smiZ];
        }
        free(iter);
        graph_toggle_neighbourhood(self, qbit);
        graph_isolate_qbit(self, qbit);

    }
    else
    {
        self->vops[qbit] = vop_lookup_table[self->vops[qbit]][projected_vop[4]];
        ll_iter_t * iter = ll_iter_t_new(self->lists[qbit]);
        npy_intp neighbour;
        while(ll_iter_next(iter, &neighbour))
        {
            self->vops[neighbour] = vop_lookup_table[self->vops[neighbour]][VOP_siZ];
        }
        free(iter);
        graph_toggle_neighbourhood(self, qbit);
        graph_isolate_qbit(self, qbit);
    }
    return 0;
}

int
graph_update_after_Z_measurement(RawGraphState * self
                            , npy_intp qbit
                            , npy_intp result)
{
    if(!result)
    {
        // Measured a +1
        // Now upating the graph state is simple:
        // U_zplus is just the identity (the rest of the graph is not changed)
        // and the vertex operator must accout for stabilizing in Z.
        // That means that K_g^(i) is changed to Z, so right multiply
        // an operator O to the VOP s.t OK_g^(i)o^\dagger = Z. That is achieved
        // by O = H.

        self->vops[qbit] = vop_lookup_table[self->vops[qbit]][projected_vop[0]];
        graph_isolate_qbit(self, qbit);

        return 0;
    }
    // Here we have to apply U_zminus to the VOP-free G-state which means
    // right multiplying that operator to the VOPs.
    self->vops[qbit] = vop_lookup_table[self->vops[qbit]][projected_vop[3]];
    ll_iter_t * iter = ll_iter_t_new(self->lists[qbit]);
    npy_intp neighbour;
    while(ll_iter_next(iter, &neighbour))
    {
        self->vops[neighbour] = vop_lookup_table[self->vops[neighbour]][VOP_Z];
    }
    free(iter);
    graph_isolate_qbit(self, qbit);

    return 0;
}


int
graph_update_after_measurement(RawGraphState * self
                            , npy_uint8 observable
                            , npy_intp qbit
                            , npy_intp result)
{
    // X measurement. Note that the qbit has neighbours.
    if(observable == 2)
    {
        return graph_update_after_X_measurement(self, qbit, result);
    }
    if(observable == 1)
    {
        return graph_update_after_Y_measurement(self, qbit, result);
    }
    return graph_update_after_Z_measurement(self, qbit, result);
}

int 
graph_can_clear_vop(RawGraphState * self, npy_intp i, npy_intp j)
{
    if(ll_has_value(self->lists[i], j))
    {
        if(ll_length(self->lists[i]) > 1)
        {
            return 1;
        }
        return 0;
    }
    if(ll_length(self->lists[i]) > 0)
    {
        return 1;
    }
    return 0;
}
