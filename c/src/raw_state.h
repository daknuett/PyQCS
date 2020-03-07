#ifndef raw_state_h_
#define raw_state_h_
#include <stdlib.h>
#include "graph_operations.h"

typedef struct
{
    long int qbit;
    long int result;
} GQCS_measurement_result;

static int
RawGraphState_init(RawGraphState * self
            , long int length
            );
static RawGraphState *
RawGraphState_deepcopy(RawGraphState * self);

static int
RawGraphState_apply_C_L(RawGraphState * self
                        , PyObject * args);

static GQCS_measurement_result *
RawGraphState_measure(RawGraphState * self, long int qbit, double random);


static void
RawGraphState_dealloc(RawGraphState * self);

static int
RawGraphState_apply_CZ(RawGraphState * self, long int i, long int j);
#endif
