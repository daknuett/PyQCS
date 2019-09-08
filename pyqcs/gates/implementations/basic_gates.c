#include <Python.h>
#include <structmember.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>
#include <stddef.h>
#include <math.h>

#include "generic_setup.h"

typedef struct
{
    npy_intp act;
    npy_intp control;
    npy_float r;
} basic_gate_argument_t;

long int ipow(int base
            , int exponent)
{
    long int result;

    // Actually most of our powers will be
    // base 2, so handle this explicitly.
    // Should be way faster.
    if(base == 2)
    {
        result = 1;
        result <<= exponent;
        return result;
    }
    // Sometimes we have base -1 powers;
    // these too, are easy to compute.
    if(base == -1)
    {
        if(exponent & 0b1)
        {
            return -1;
        }
        return 1;
    }
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

    basic_gate_argument_t argument = *((basic_gate_argument_t *) data);
    npy_intp i;
    PYQCS_GATE_GENERIC_SETUP;

    for(i = 0; i < ndim; i++)
    {
        qm_out[i] = qm_in[i ^ (1 << argument.act)];
    }
    *measured_out = 0;
}

static void
ufunc_R( char ** args
       , npy_intp * dimensions
       , npy_intp * steps
       , void * data)
{

    basic_gate_argument_t argument = *((basic_gate_argument_t *) data);
    npy_intp i;
    PYQCS_GATE_GENERIC_SETUP;

    for(i = 0; i < ndim; i++)
    {
        if(i & (1 << argument.act))
        {
            qm_out[i].real = cos(argument.r) * qm_in[i].real;
            qm_out[i].imag = sin(argument.r) * qm_in[i].imag;
        }
        else
        {
            qm_out[i] = qm_in[i];
        }
    }
    *measured_out = 0;
}

static void
ufunc_H( char ** args
       , npy_intp * dimensions
       , npy_intp * steps
       , void * data)
{
    basic_gate_argument_t argument = *((basic_gate_argument_t *) data);
    PYQCS_GATE_GENERIC_SETUP;
    npy_intp i;

    for(i = 0; i < ndim; i++)
    {
        npy_intp not_index = i ^ (1 << argument.act);

        npy_intp sign;
        if(!(i & (2 << argument.act))
            && ipow(-1, not_index) == -1)
        {
            sign = -1;
        }
        else
        {
            sign = 1;
        }

        
        qm_out[i].real = (qm_in[i].real 
                             + sign * qm_in[not_index].real) 
                         * M_SQRT1_2;
        qm_out[i].imag = (qm_in[i].imag 
                             + sign * qm_in[not_index].imag) 
                         * M_SQRT1_2;
    }

    *measured_out = 0;
}

static char ufunc_types[5] = 
    { NPY_CDOUBLE, NPY_DOUBLE, NPY_CDOUBLE, NPY_DOUBLE, NPY_DOUBLE };
static PyUFuncGenericFunction ufunc_X_funcs[1] = 
    { ufunc_X };
static PyUFuncGenericFunction ufunc_H_funcs[1] = 
    { ufunc_H };
static PyUFuncGenericFunction ufunc_R_funcs[1] = 
    { ufunc_R };


typedef struct
{
    PyObject_HEAD
    basic_gate_argument_t argument;
    PyObject * ufunc;
    void * data[1];
} BasicGate;


static int
BasicGate_init
	( BasicGate * self
	, PyObject * args)
{
	char type;

    //Py_INCREF(args);
	if(!PyArg_ParseTuple(args, "Clld"
                , &type
                , &(self->argument.act)
                , &(self->argument.control)
                , &(self->argument.r)))
	{
		return -1;
	}
    
	self->data[0] = (void *)(&(self->argument));

	switch(type)
	{
		case 'X':
		{
			self->ufunc = PyUFunc_FromFuncAndDataAndSignature(
				ufunc_X_funcs // func
				, self->data // data
				, ufunc_types //types
				, 1 // ntypes
				, 2 // nin
				, 3 // nout
				, PyUFunc_None // identity
				, "X_function" // name
				, "Computes the X (NOT) gate on a state." // doc
				, 0 // unused
                , "(n),(m)->(n),(m),()"); 

            if(self->ufunc <= 0)
            {
                //I have no idea what is going on.
                //PyErr_SetString(PyExc_ValueError, "failed to construct the ufunc for unknow reasons");
                return -1;
            }
			break;
		}
		case 'H':
		{
			self->ufunc = PyUFunc_FromFuncAndDataAndSignature(
				ufunc_H_funcs // func
				, self->data // data
				, ufunc_types //types
				, 1 // ntypes
				, 2 // nin
				, 3 // nout
				, PyUFunc_None // identity
				, "H_function" // name
				, "Computes the H (Hadamard) gate on a state." // doc
				, 0 // unused
                , "(n),(m)->(n),(m),()"); 

            if(self->ufunc <= 0)
            {
                //I have no idea what is going on.
                //PyErr_SetString(PyExc_ValueError, "failed to construct the ufunc for unknow reasons");
                return -1;
            }
			break;
		}
		case 'R':
		{
			self->ufunc = PyUFunc_FromFuncAndDataAndSignature(
				ufunc_R_funcs // func
				, self->data // data
				, ufunc_types //types
				, 1 // ntypes
				, 2 // nin
				, 3 // nout
				, PyUFunc_None // identity
				, "R_function" // name
				, "Computes the R (rotation) gate on a state." // doc
				, 0 // unused
                , "(n),(m)->(n),(m),()"); 

            if(self->ufunc <= 0)
            {
                //I have no idea what is going on.
                //PyErr_SetString(PyExc_ValueError, "failed to construct the ufunc for unknow reasons");
                return -1;
            }
			break;
		}
        default:
        {
            PyErr_SetString(PyExc_ValueError, "Type must be one of X,H,R,C,M");
            return -1;
        }
    }
    Py_INCREF(self->ufunc);
	return 0;
}

static PyObject *
BasicGate_call
	(BasicGate * self
	 , PyObject * args
	 , PyObject * kwargs)
{
	return PyObject_Call(self->ufunc, args, kwargs);
}

static void BasicGate_dealloc(BasicGate * self)
{
    Py_XDECREF(self->ufunc);
    Py_TYPE(self)->tp_free((PyObject *) self);
}


static PyMemberDef BasicGate_members[] = 
{
	{"ufunc", T_OBJECT_EX, offsetof(BasicGate, ufunc), 0, "ufunc"},
	{NULL}
};


static PyTypeObject BasicGateType = 
{
	PyVarObject_HEAD_INIT(NULL, 0)
	.tp_name = "pyqcs.gates.implementations.basic_gates.BasicGate",
	.tp_doc = "The wrapper for the basic gates X, H, R_phi, CNOT and Measurement." \
              "The first argument is the type of the gate as a char: X,H,R,C,M." 
               ,
	.tp_basicsize = sizeof(BasicGate),
	.tp_itemsize = 0,
	.tp_flags = Py_TPFLAGS_DEFAULT,
	.tp_new = PyType_GenericNew,
	.tp_init = (initproc) BasicGate_init,
	.tp_call = BasicGate_call,
	.tp_members = BasicGate_members,
    .tp_dealloc = (destructor) BasicGate_dealloc
};



static PyMethodDef basic_gates_methods[] = {
	{NULL, NULL, 0, NULL}
};


static struct PyModuleDef moduledef = {
	PyModuleDef_HEAD_INIT
	, "pyqcs.gates.implementations.basic_gates"
	, NULL
	, -1
	, basic_gates_methods
	, NULL
	, NULL
	, NULL
	, NULL
};

PyMODINIT_FUNC 
PyInit_basic_gates(void)
{
	PyObject * module;

	if(PyType_Ready(&BasicGateType) < 0)
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

	Py_INCREF(&BasicGateType);
	PyModule_AddObject(module, "BasicGate", (PyObject *) &BasicGateType);

	return module;
}

