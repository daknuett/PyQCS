#include <Python.h>
#include <numpy/ufunc.h>
#include "generic_setup.h"


long int ipow(int base
            , int exponent)
{

    long int result;
    if(exponent == 0)
    {
        return 1;
    }
    if(exponent % 2 == 0)
    {
        result = ipow(base, exponent / 2);
        return result * result;
    }
    return base * ipow(base, exponent - 1);
                
}


static void
ufunc_X( char ** args
       , npy_intp * dimensions
       , npy_intp * steps
       , void * data)
{

    npy_intp l = *((npy_intp *) data);
    npy_intp i;
    PYQCS_GATE_GENERIC_SETUP;

    for(i = 0; i < ndim; i++)
    {
        qm_out[i] = qm_in[i^ipow(2, l)];
    }
    *measured_out = 0;
}
