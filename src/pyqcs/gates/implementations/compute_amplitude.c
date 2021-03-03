#include <Python.h>
#include <structmember.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>
#include <stddef.h>
#include <math.h>

static void
ufunc_compute_amplitude(
        char ** args
       , npy_intp * dimensions
       , npy_intp * steps
       , void * data)
{
    npy_cdouble * qm_in = (npy_cdouble *)(args[0]);
    npy_intp * qbits_in = (npy_intp *)(args[1]);
    npy_uint8 * bitstr_in = (npy_uint8 *)(args[2]);
    npy_double * amplitude = (npy_double *)(args[3]);

    npy_intp ndims = dimensions[1];
    npy_intp nqbits = dimensions[2];

    npy_intp i;
    npy_uint64 check_bits = 0;
    npy_uint64 bit_mask = 0;
    for(i = 0; i < nqbits; i++)
    {
        // This is how it is supposed to work.
        // I don't know why it doesn't.
        //check_bits |= 1 << *(qbits_in + i*steps[1]);
        //if(*(bitstr_in + i*steps[2]))
        //{
        //    bit_mask |= 1 << *(qbits_in + i*steps[1]);
        //}
        check_bits |= 1 << qbits_in[i];
        if(bitstr_in[i])
        {
            bit_mask |= 1 << qbits_in[i];
        }
    }

    *amplitude = 0;

    for(i = 0; i < ndims; i++)
    {
        npy_uint64 bits_that_matter = i & check_bits;
        if((bits_that_matter ^ bit_mask) == 0)
        {
            *amplitude += qm_in[i].real*qm_in[i].real + qm_in[i].imag*qm_in[i].imag;
        }
    }

}

static char ufunc_compute_amplitude_types[4] = 
{
    NPY_CDOUBLE, NPY_INTP, NPY_UINT8, NPY_DOUBLE
};
static PyUFuncGenericFunction ufunc_compute_amplitude_funcs[1] =
{
    ufunc_compute_amplitude
};
static PyMethodDef compute_amplitude_methods[] = {
	{NULL, NULL, 0, NULL}
};
static struct PyModuleDef moduledef = {
	PyModuleDef_HEAD_INIT
	, "pyqcs.gates.implementations.compute_amplitude"
	, NULL
	, -1
	, compute_amplitude_methods
	, NULL
	, NULL
	, NULL
	, NULL
};

PyMODINIT_FUNC 
PyInit_compute_amplitude(void)
{
    char * data = "NULL";
	import_array();
	import_ufunc();
	PyObject * module;
    PyObject * compute_amplitude = PyUFunc_FromFuncAndDataAndSignature(
				ufunc_compute_amplitude_funcs // func
                // It seems that data must not be NULL
                // in contrast to what https://numpy.org/devdocs/reference/c-api/ufunc.html
                // says.
				, (void **)&data // data
                //, NULL
				, ufunc_compute_amplitude_types //types
				, 1 // ntypes
				, 3 // nin
				, 1 // nout
				, PyUFunc_None // identity
				, "compute_amplitude" // name
				, "Computes the amplitude of the given bitstr on the given qbits." // doc
				, 0 // unused
                , "(n),(m),(m)->()"); 
    if(compute_amplitude <= 0)
    {
        return NULL;
    }
	module = PyModule_Create(&moduledef);
	if(!module)
	{
		return NULL;
	}
    Py_INCREF(compute_amplitude);

    if(PyModule_AddObject(module, "compute_amplitude", compute_amplitude) < 0)
    {
        Py_XDECREF(compute_amplitude);
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
