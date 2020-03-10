#include <stdio.h>
#include "exec.h"
#include "error.h"


int
read_next_instruction(GQCS_bytecode_instruction_t * out, FILE * in)
{
    size_t read;

    read = fread(out, sizeof(GQCS_bytecode_instruction_t), 1, in);

    if(read != 1)
    {
        GQCS_set_error("EOF reached before instruction was complete");
        return -1;
    }

    if(!GQCS_bytecode_instruction_valid(out))
    {
        GQCS_set_error("instruction invalid");
        return -2;
    }
    return 0;
}

int
exec_instruction(RawGraphState * self
        , GQCS_bytecode_instruction_t * instruction)
{
    switch(instruction->command)
    {
        case 'L':
            {
                return RawGraphState_apply_C_L(self
                                        , instruction->act
                                        , (uint8_t) instruction->argument);
            }
        case 'Z':
            {
                return RawGraphState_apply_CZ(self
                                        , instruction->act
                                        , instruction->argument);
            }
        case 'M':
            {
                GQCS_measurement_result * result;
                double r = random() / (1.0 * RAND_MAX);
                result = RawGraphState_measure(self, instruction->act, r);
                if(!result)
                {
                    return -1;
                }
                free(result);
                return 0;
            }
        default:
            {
                GQCS_set_error("internal error: invalid instruction encountered");
                return -1;
            }
    }
}

int
exec_all(RawGraphState * self
        , FILE * in)
{
    GQCS_bytecode_instruction_t * instruction = malloc(sizeof(GQCS_bytecode_instruction_t));
    int result;
    while(1)
    {

        result = read_next_instruction(instruction, in);
        if(result < -1)
        {
            free(instruction);
            return -1;
        }
        if(result < 0)
        {
            free(instruction);
            return 0;
        }

        if(exec_instruction(self, instruction))
        {
            free(instruction);
            return -1;
        }
    }
}
