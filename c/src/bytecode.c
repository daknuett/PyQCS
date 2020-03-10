#include <stdlib.h>
#include <stdio.h>
#include "bytecode.h"
#include "error.h"



int
GQCS_bytecode_head_valid(GQCS_bytecode_head_t * head)
{
    if( head->magic_word[0] != 'G'
        || head->magic_word[1] != 'Q'
        || head->magic_word[2] != 'C'
        || head->magic_word[3] != 'S')
    {
        GQCS_set_error("header magic word invalid");
        return 0;
    }

    if( head->expect_b != 'b'
        || head->expect_s != 's'
        || head->expect_q != 'q')
    {
        GQCS_set_error("found garbage data in header");
        return 0;
    }

    return 1;
}

GQCS_full_header_t * 
GQCS_bytecode_read_full_header(GQCS_bytecode_head_t * head
                                , FILE * fin)
{
    uint64_t * samples = malloc((head->nsampleqbits + 1) * sizeof(uint64_t));
    size_t read = fread(samples, sizeof(uint64_t), head->nsampleqbits + 1, fin);
    if(read != head->nsampleqbits + 1)
    {
        free(samples);
        GQCS_set_error("failed to read sample qbits");
        return NULL;
    }

    if(samples[head->nsampleqbits] != 0xffffffffffffffff)
    {
        free(samples);
        GQCS_set_error("failed to validate header");
        return NULL;
    }

    GQCS_full_header_t * header = malloc(sizeof(GQCS_bytecode_head_t));
    header->head = head;
    header->sampleqbits = samples;
    return header;
}

void
GQCS_bytecode_full_header_dealloc(GQCS_full_header_t * header)
{
    free(header->head);
    free(header->sampleqbits);
    free(header);
}

int
GQCS_bytecode_instruction_valid(GQCS_bytecode_instruction_t * instruction)
{
    if(!( instruction->command == 'L'
        || instruction->command == 'Z'
        || instruction->command == 'M'
        ))
    {
        GQCS_set_error("invalid command");
        return 0;
    }
    if(instruction->command == 'M'
        && instruction->argument != 0)
    {
        GQCS_set_error("found garbage data in measurement command");
        return 0;
    }
    return 1;
}
