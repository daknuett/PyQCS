#ifndef linked_list_h_
#define linked_list_h_

typedef struct ll_node_s
{
    struct ll_node_s * next;
    long int value;
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
ll_iter_next(ll_iter_t * iter, long int * result);

// Resets the iterator. Always returns 0.
int
ll_iter_reset(ll_iter_t * iter);

void
ll_recursively_delete_list(ll_node_t ** list);

ll_node_t *
ll_node_t_new(ll_node_t * next, long int value);

int
ll_insert_value(ll_node_t ** list, long int value);

int
ll_delete_value(ll_node_t ** list, long int value);

int
ll_has_value(ll_node_t * list, long int value);

long int
ll_length(ll_node_t * list);

int
ll_deepcopy(ll_node_t ** destination, ll_node_t ** source);

#endif
