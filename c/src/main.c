#include <stdio.h>
#include <stdlib.h>

#include "raw_state.h"
#include "error.h"
#include "bytecode.h"
#include "exec.h"
#include "sample.h"


int main(int argc, char ** argv)
{
    int result = 0;
    if(argc < 2)
    {
        fprintf(stderr, "FATAL: missing bytecode file\n");
        return -1;
    }

    FILE * out = stdout;
    int close_out = 0;

    FILE * bytecode_file;

    if(argc > 2)
    {
        out = fopen(argv[2], "w");
        if(!out)
        {
            fprintf(stderr, "FATAL: failed to open %s for writing\n", argv[2]);
            return -1;
        }
        close_out = 1;
    }

    bytecode_file = fopen(argv[1], "r");
    if(!bytecode_file)
    {
        fprintf(stderr, "FATAL: failed to open %s for reading\n", argv[1]);
        goto exit;
    }

    // Read the header from the given
    // bytecode file.
    
    GQCS_bytecode_head_t * head = malloc(sizeof(GQCS_bytecode_head_t));
    if(fread(head, sizeof(GQCS_bytecode_head_t), 1, bytecode_file) != 1)
    {
        fprintf(stderr, "FATAL: failed to read the head from %s\n", argv[1]);
        free(head);
        result = -1;
        goto exit;
    }
    
    if(!GQCS_bytecode_head_valid(head))
    {
        fprintf(stderr, "SYSTEM: %s\n", GQCS_err_msg);
        fprintf(stderr, "FATAL: invalid head.\n");
        free(head);
        result = -1;
        goto exit;
    }

    GQCS_full_header_t * header = GQCS_bytecode_read_full_header(head, bytecode_file);
    if(!header)
    {
        fprintf(stderr, "SYSTEM: %s\n", GQCS_err_msg);
        fprintf(stderr, "FATAL: invalid header.\n");
        free(head);
        result = -1;
        goto exit;
    }

    RawGraphState * state = malloc(sizeof(RawGraphState));
    if(RawGraphState_init(state, head->nqbits) < 0)
    {
        fprintf(stderr, "FATAL: failed to initialize state.\n");
        GQCS_bytecode_full_header_dealloc(header);
        free(state);
        result = -1;
        goto exit;
    }


    if(exec_all(state, bytecode_file))
    {
        fprintf(stderr, "SYSTEM: %s\n", GQCS_err_msg);
        fprintf(stderr, "FATAL: error while executing byte code.\n");
        GQCS_bytecode_full_header_dealloc(header);
        RawGraphState_dealloc(state);
        result = -1;
        goto exit;
    }

    print_sampling_head(header, out);

    if(do_and_print_samples(state, header, out))
    {
        fprintf(stderr, "SYSTEM: %s\n", GQCS_err_msg);
        fprintf(stderr, "FATAL: error while sampling results.\n");
        GQCS_bytecode_full_header_dealloc(header);
        RawGraphState_dealloc(state);
        result = -1;
        goto exit;
    }

    GQCS_bytecode_full_header_dealloc(header);
    RawGraphState_dealloc(state);

exit:
    fclose(bytecode_file);
    if(close_out)
    {
        fclose(out);
    }
    return result;
}
