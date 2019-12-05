#include "linked_list.h"

void
ll_recursively_delete_list(ll_node_t ** list)
{
    ll_node_t * next_node;
    while(*list)
    {
        next_node = (*list)->next;
        free(*list);
        *list = next_node;
    }
    *list = NULL;
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

npy_intp
ll_length(ll_node_t * list)
{
    npy_intp result = 0;
    while(list)
    {
        result ++;
        list = list->next;
    }
    return result;
}


ll_iter_t * 
ll_iter_t_new(ll_node_t * list)
{
    ll_iter_t * result = malloc(sizeof(ll_iter_t));
    result->start = list;
    result->current = list;
    return result;
}

int
ll_iter_next(ll_iter_t * iter, npy_intp * result)
{
    if(!iter->current)
    {
        return 0;
    }
    *result = iter->current->value;
    iter->current = iter->current->next;
    return 1;
}

int
ll_iter_reset(ll_iter_t * iter)
{
    iter->current = iter->start;
    return 0;
}

int
ll_deepcopy(ll_node_t ** destination, ll_node_t ** source)
{
    ll_node_t * current_source = *source;
    ll_node_t * new_node = NULL;
    ll_node_t * last_node = NULL;

    while(current_source)
    {
        new_node = ll_node_t_new(NULL, current_source->value);
        if(!new_node)
        {
            return -1;
        }
        current_source = current_source->next;
        if(!last_node)
        {
            *destination = new_node;
        }
        else
        {
            last_node->next = new_node;
        }
        last_node = new_node;
        new_node = NULL;
    }
    return 0;
}
