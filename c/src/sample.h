#ifndef GQCS_sample_h_
#define GQCS_sample_h_

#include "raw_state.h"
#include "bytecode.h"

int
measure_one_qbit(long int * result
        , RawGraphState * state
        , long int qbit);

int
print_sampling_head(GQCS_full_header_t * header
                    , FILE * out);

int
do_and_print_one_series(RawGraphState * state
                    , GQCS_full_header_t * header
                    , long int seriesno
                    , FILE * out);

int
do_and_print_samples(RawGraphState * state
                    , GQCS_full_header_t * header
                    , FILE * out);


#endif
