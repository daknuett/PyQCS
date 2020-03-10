#ifndef raw_state_h_
#define raw_state_h_
#include <stdlib.h>
#include "vops.h"
#include "graph_operations.h"

typedef struct
{
    long int qbit;
    long int result;
} GQCS_measurement_result;

int
RawGraphState_init(RawGraphState * self
            , long int length
            );
RawGraphState *
RawGraphState_deepcopy(RawGraphState * self);

int
RawGraphState_apply_C_L(RawGraphState * self
                        , long int i
                        , uint8_t vop);

GQCS_measurement_result *
RawGraphState_measure(RawGraphState * self, long int qbit, double random);


void
RawGraphState_dealloc(RawGraphState * self);

int
RawGraphState_apply_CZ(RawGraphState * self, long int i, long int j);
#endif
