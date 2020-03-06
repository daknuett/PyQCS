#ifndef linked_list_h_
#define linked_list_h_
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>

typedef struct ll_node_s
{
    struct ll_node_s * next;
    npy_intp value;
} ll_node_t;

typedef struct ll_iter_s
{
    ll_node_t * start;
    ll_node_t * current;
} ll_iter_t;

ll_iter_t * 
ll_iter_t_new(ll_node_t * list);


// Returns 1 as long as there are elements that have
// not been iterated yet and stores the current value in result.
int
ll_iter_next(ll_iter_t * iter, npy_intp * result);

// Resets the iterator. Always returns 0.
int
ll_iter_reset(ll_iter_t * iter);

void
ll_recursively_delete_list(ll_node_t ** list);

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

int
ll_deepcopy(ll_node_t ** destination, ll_node_t ** source);

#endif
