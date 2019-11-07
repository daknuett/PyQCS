#ifndef linked_list_h_
#define linked_list_h_
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>

typedef struct ll_node_s
{
    struct ll_node_s * next;
    npy_intp value;
} ll_node_t;

void
ll_recursively_delete_list(ll_node_t * list);

ll_node_t *
ll_node_t_new(ll_node_t * next, npy_intp value);

int
ll_insert_value(ll_node_t ** list, npy_intp value);

int
ll_delete_value(ll_node_t ** list, npy_intp value);

int
ll_has_value(ll_node_t * list, npy_intp value);

npy_intp
ll_length(ll_node_t * list);

#endif
