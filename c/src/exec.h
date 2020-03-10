#ifndef GQCS_exec_h_
#define GQCS_exec_h_

#include <stdio.h>

#include "bytecode.h"
#include "raw_state.h"

int
read_next_instruction(GQCS_bytecode_instruction_t * out, FILE * in);

int
exec_instruction(RawGraphState * self
        , GQCS_bytecode_instruction_t * instruction);

int
exec_all(RawGraphState * self
        , FILE * in); 
#endif
