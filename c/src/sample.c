#include <stdlib.h>

#include "sample.h"
#include "error.h"


int
measure_one_qbit(long int * result
        , RawGraphState * state
        , long int qbit)
{
    GQCS_measurement_result * mres;
    double r = random() / (1.0 * RAND_MAX);

    mres = RawGraphState_measure(state, qbit, r);

    if(!mres)
    {
        return -1;
    }

    *result = mres->result;

    free(mres);
    return 0;
}

int
print_sampling_head(GQCS_full_header_t * header
                    , FILE * out)
{
    long int i;

    fprintf(out, "  ");
    
    for(i = 0; i < header->head->nsampleqbits; i++)
    {
        fprintf(out, ", %ld", header->sampleqbits[i]);
    }
    fprintf(out, "\n");
    return 0;
}

int
do_and_print_one_series(RawGraphState * state
                    , GQCS_full_header_t * header
                    , long int seriesno
                    , FILE * out)
{
    long int i;
    long int result;

    RawGraphState * this_state = RawGraphState_deepcopy(state);
    if(!this_state)
    {
        GQCS_set_error("failed to copy state for measurement");
        return -1;
    }

    fprintf(out, "%ld", seriesno);

    for(i = 0; i < header->head->nsampleqbits; i++)
    {
        if(measure_one_qbit(&result, this_state, header->sampleqbits[i]))
        {
            RawGraphState_dealloc(this_state);
            return -1;
        }

        fprintf(out, ", %ld", result);
    }
    fprintf(out, "\n");
    RawGraphState_dealloc(this_state);
    return 0;
}

int
do_and_print_samples(RawGraphState * state
                    , GQCS_full_header_t * header
                    , FILE * out)
{
    int i;
    
    for(i = 0; i < header->head->nsamples; i++)
    {
        if(do_and_print_one_series(state, header, i, out))
        {
            return -1;
        }
    }
    return 0;
}

