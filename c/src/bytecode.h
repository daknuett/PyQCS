#ifndef GQCS_bytecode_h_
#define GQCS_bytecode_h_

#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>

typedef struct __attribute__((packed))
{
    uint8_t magic_word[4];
    uint64_t nqbits;
    uint8_t expect_b;
    uint16_t nsamples;
    uint8_t expect_s;
    uint64_t nsampleqbits;
    uint8_t expect_q;
} GQCS_bytecode_head_t;

int
GQCS_bytecode_head_valid(GQCS_bytecode_head_t * head);

typedef struct
{
    GQCS_bytecode_head_t * head;
    uint64_t * sampleqbits;
} GQCS_full_header_t;

GQCS_full_header_t * 
GQCS_bytecode_read_full_header(GQCS_bytecode_head_t * head
                                , FILE * fin);
void
GQCS_bytecode_full_header_dealloc(GQCS_full_header_t * header);

typedef struct __attribute__((packed))
{
    uint8_t command;
    uint64_t act;
    uint64_t argument;
} GQCS_bytecode_instruction_t;

int
GQCS_bytecode_instruction_valid(GQCS_bytecode_instruction_t * instruction);

#endif
