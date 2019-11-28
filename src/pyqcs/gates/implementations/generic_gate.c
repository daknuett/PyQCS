#include <Python.h>
#include <structmember.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>
#include <stddef.h>
#include <math.h>

#include "generic_setup.h"

void
copy_cl_state(npy_int8 * new, npy_int8 * old, npy_intp nbits)
{
    npy_intp i;
    for(i = 0; i < nbits; i++)
    {
        new[i] = old[i];
    }
}

typedef struct
{
    npy_intp act;
    Py_complex u2u;
    Py_complex d2d;
    Py_complex d2u;
    Py_complex u2d;
} generic_gate_argument_t;


static void
ufunc( char ** args
       , npy_intp * dimensions
       , npy_intp * steps
       , void * data)

{
    generic_gate_argument_t argument = *((generic_gate_argument_t *)data);

    npy_intp i;
    PYQCS_GATE_GENERIC_SETUP;

    copy_cl_state(cl_out, cl_in, nqbits);

    for(i = 0; i < ndim; i++)
    {
        npy_intp not_index = i ^ (1 << argument.act);

        if(!(i & (1 << argument.act)))
        {
            qm_out[i].real =  qm_in[i].real * argument.u2u.real 
                            + qm_in[not_index].real * argument.d2u.real 
                            - qm_in[i].imag * argument.u2u.imag 
                            - qm_in[not_index].imag * argument.d2u.imag;
            qm_out[i].imag =  qm_in[i].imag * argument.u2u.real 
                            + qm_in[not_index].imag * argument.d2u.real 
                            + qm_in[i].real * argument.u2u.imag 
                            + qm_in[not_index].real * argument.d2u.imag;
        }
        else
        {
            qm_out[i].real =  qm_in[i].real * argument.d2d.real 
                            + qm_in[not_index].real * argument.u2d.real 
                            - qm_in[i].imag * argument.d2d.imag 
                            - qm_in[not_index].imag * argument.u2d.imag;
            qm_out[i].imag =  qm_in[i].imag * argument.d2d.real 
                            + qm_in[not_index].imag * argument.u2d.real 
                            + qm_in[i].real * argument.d2d.imag 
                            + qm_in[not_index].real * argument.u2d.imag;
        }
    }
    *measured_out = 0;
}

static char ufunc_types[5] = 
    { NPY_CDOUBLE, NPY_INT8, NPY_CDOUBLE, NPY_INT8, NPY_UINT64 };
static PyUFuncGenericFunction ufunc_funcs[1] = 
    { ufunc };

typedef struct
{
    PyObject_HEAD
    generic_gate_argument_t argument;
    PyObject * ufunc;
    void * data[1];
} GenericGate;


static int
GenericGate_init
	( GenericGate * self
	, PyObject * args)
{
	if(!PyArg_ParseTuple(args, "lDDDD"
                , &(self->argument.act)
                , &(self->argument.u2u)
                , &(self->argument.d2d)
                , &(self->argument.d2u)
                , &(self->argument.u2d)
                )
            )
	{
		return -1;
	}

    self->data[0] = (void*)(&(self->argument));

    self->ufunc = PyUFunc_FromFuncAndDataAndSignature(
        ufunc_funcs// func
        , self->data // data
        , ufunc_types //types
        , 1 // ntypes
        , 2 // nin
        , 3 // nout
        , PyUFunc_None // identity
        , "X_function" // name
        , "Computes a generic gate on a state." // doc
        , 0 // unused
        , "(n),(m)->(n),(m),()"); 
    if(self->ufunc == 0)
    {
        //I have no idea what is going on.
        //PyErr_SetString(PyExc_ValueError, "failed to construct the ufunc for unknow reasons");
        return -1;
    }

    Py_INCREF(self->ufunc);
    return 0;
}


static PyObject *
GenericGate_call
	(GenericGate * self
	 , PyObject * args
	 , PyObject * kwargs)
{
	return PyObject_Call(self->ufunc, args, kwargs);
}

static void 
GenericGate_dealloc
    (GenericGate * self)
{
    Py_XDECREF(self->ufunc);
    Py_TYPE(self)->tp_free((PyObject *) self);
}


static PyMemberDef GenericGate_members[] = 
{
	{"ufunc", T_OBJECT_EX, offsetof(GenericGate, ufunc), 0, "ufunc"},
	{NULL}
};


static PyTypeObject GenericGateType =
{
	PyVarObject_HEAD_INIT(NULL, 0)
	.tp_name = "pyqcs.gates.implementations.generic_gate.GenericGate",
	.tp_doc = "The wrapper for the generic gates",
	.tp_basicsize = sizeof(GenericGate),
	.tp_itemsize = 0,
	.tp_flags = Py_TPFLAGS_DEFAULT,
	.tp_new = PyType_GenericNew,
	.tp_init = (initproc) GenericGate_init,
	.tp_call = GenericGate_call,
	.tp_members = GenericGate_members,
    .tp_dealloc = (destructor) GenericGate_dealloc
};



static PyMethodDef generic_gate_methods[] = {
	{NULL, NULL, 0, NULL}
};


static struct PyModuleDef moduledef = {
	PyModuleDef_HEAD_INIT
	, "pyqcs.gates.implementations.generic_gate"
	, NULL
	, -1
	, generic_gate_methods
	, NULL
	, NULL
	, NULL
	, NULL
};

PyMODINIT_FUNC 
PyInit_generic_gate(void)
{
	PyObject * module;
    //static void * generic_gate_API[generic_gate_API_pointers];
    //PyObject * api_obj;

	if(PyType_Ready(&GenericGateType) < 0)
	{
		return NULL;
	}

	module = PyModule_Create(&moduledef);
	if(!module)
	{
		return NULL;
	}
	import_array();
	import_ufunc();

	Py_INCREF(&GenericGateType);
	if(PyModule_AddObject(module, "GenericGate", (PyObject *) &GenericGateType) < 0)
    {
        Py_XDECREF(&GenericGateType);
        Py_DECREF(module);
        return NULL;
    }

    //generic_gate_API[0] = (void *)&GenericGateType;
    //api_obj = PyCapsule_New((void *) generic_gate_API, "pyqcs.gates.implementations.generic_gate._C_API", NULL);
    //
    //if(PyModule_AddObject(module, "_C_API", api_obj) < 0)
    //{
    //    Py_DECREF(module);
    //    Py_XDECREF(api_obj);
    //    return NULL;
    //}


	return module;
}

