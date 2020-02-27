#include <Python.h>
#include <structmember.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>
#include <stddef.h>
#include <math.h>

#include "generic_setup.h"

#define basic_gates_module
#include "basic_gates.h"

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
    npy_intp control;
    npy_double r;
    PyObject * rng;
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

    copy_cl_state(cl_out, cl_in, nqbits);

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

    copy_cl_state(cl_out, cl_in, nqbits);

    for(i = 0; i < ndim; i++)
    {
        if(i & (1 << argument.act))
        {
            qm_out[i].real = cos(argument.r)*qm_in[i].real - sin(argument.r)*qm_in[i].imag;
            qm_out[i].imag = sin(argument.r)*qm_in[i].real + cos(argument.r)*qm_in[i].imag;
        }
        else
        {
            qm_out[i] = qm_in[i];
        }
    }
    *measured_out = 0;
}

static void
ufunc_Z( char ** args
       , npy_intp * dimensions
       , npy_intp * steps
       , void * data)
{

    basic_gate_argument_t argument = *((basic_gate_argument_t *) data);
    npy_intp i;
    PYQCS_GATE_GENERIC_SETUP;

    copy_cl_state(cl_out, cl_in, nqbits);

    for(i = 0; i < ndim; i++)
    {
        if(i & (1 << argument.act))
        {
            qm_out[i].real = -1 * qm_in[i].real;
            qm_out[i].imag = -1 * qm_in[i].imag;
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

    copy_cl_state(cl_out, cl_in, nqbits);

    for(i = 0; i < ndim; i++)
    {
        npy_intp not_index = i ^ (1 << argument.act);


        if(!(i & (1 << argument.act)))
        {
            // This is the |1> state. Just add up.
            qm_out[i].real = (qm_in[i].real + qm_in[not_index].real) * M_SQRT1_2;
            qm_out[i].imag = (qm_in[i].imag + qm_in[not_index].imag) * M_SQRT1_2;
        }
        else
        {
            // This is the |0> state. Subtract the |0> amplitude from the |1> amplitude.
            qm_out[i].real = (qm_in[not_index].real - qm_in[i].real) * M_SQRT1_2;
            qm_out[i].imag = (qm_in[not_index].imag - qm_in[i].imag) * M_SQRT1_2;

        }
    }

    *measured_out = 0;
}

static void
ufunc_C( char ** args
       , npy_intp * dimensions
       , npy_intp * steps
       , void * data)
{
    basic_gate_argument_t argument = *((basic_gate_argument_t *) data);
    PYQCS_GATE_GENERIC_SETUP;
    npy_intp i;

    copy_cl_state(cl_out, cl_in, nqbits);

    for(i = 0; i < ndim; i++)
    {

        if((i & (1 << argument.control)))
        {
            qm_out[i].real = qm_in[i ^ (1 << argument.act)].real;
            qm_out[i].imag = qm_in[i ^ (1 << argument.act)].imag;
        }
        else
        {

            qm_out[i].real = qm_in[i].real;
            qm_out[i].imag = qm_in[i].imag;
        }
    }
    *measured_out = 0;
}
static void
ufunc_CZ( char ** args
       , npy_intp * dimensions
       , npy_intp * steps
       , void * data)
{
    basic_gate_argument_t argument = *((basic_gate_argument_t *) data);
    PYQCS_GATE_GENERIC_SETUP;
    npy_intp i;

    copy_cl_state(cl_out, cl_in, nqbits);

    for(i = 0; i < ndim; i++)
    {

        if((i & (1 << argument.control)) && (i & (1 << argument.act)))
        {
            qm_out[i].real = qm_in[i].real * -1;
            qm_out[i].imag = qm_in[i].imag * -1;
        }
        else
        {

            qm_out[i].real = qm_in[i].real;
            qm_out[i].imag = qm_in[i].imag;
        }
    }
    *measured_out = 0;
}


static void
ufunc_M( char ** args
       , npy_intp * dimensions
       , npy_intp * steps
       , void * data)
{
    basic_gate_argument_t argument = *((basic_gate_argument_t *) data);
    PYQCS_GATE_GENERIC_SETUP;
    npy_intp i;

    npy_double amplitude_1 = 0;
    npy_double amplitude_0 = 0;

    // TODO: this can be optimized.
    for(i = 0; i < ndim; i++)
    {
        if(i & (1 << argument.act))
        {
            // XXX: do we get big errors here?
            amplitude_1 += qm_in[i].real * qm_in[i].real;
            amplitude_1 += qm_in[i].imag * qm_in[i].imag;
        }
        else
        {
            // XXX: do we get big errors here?
            amplitude_0 += qm_in[i].real * qm_in[i].real;
            amplitude_0 += qm_in[i].imag * qm_in[i].imag;
        }
    }

    // XXX:
    // This is supposed to account for numerical errors that happened
    // in the sum above. Because the error is unknown try to estimate the 
    // error by computing the average of amplitude_0 and amplitude_1 (should be 1/2).
    // If this average is smaller than 1/2 add the missing part to the amplitudes
    // or substract it in the other case.
    
    npy_double amplitude_avg_dif = (amplitude_1 + amplitude_0) / 2 - 0.5;
    amplitude_1 -= amplitude_avg_dif;
    

    npy_double randr;
    //==================================================//
    // Get some random value. I do not like the way this
    // is done but it seems like there is no better way.
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject * random_result = PyObject_CallFunctionObjArgs(argument.rng, NULL);
    PyGILState_Release(gstate);

    if(!PyFloat_Check(random_result))
    {
        randr = 0;
    }
    else
    {
        randr = PyFloat_AsDouble(random_result);
    }
    Py_DECREF(random_result);
    //==================================================//

    copy_cl_state(cl_out, cl_in, nqbits);


    npy_double partial_amplitude;
    if(amplitude_1 >= randr)
    {
        cl_out[argument.act] = 1;
        // Measured 1; now collaps this qbit.
        partial_amplitude = 1 / sqrt(amplitude_1);
        for(i = 0; i < ndim; i++)
        {
            if(i & (1 << argument.act))
            {
                qm_out[i].real = partial_amplitude * qm_in[i].real;
                qm_out[i].imag = partial_amplitude * qm_in[i].imag;
            }
            else
            {
                qm_out[i].real = 0;
                qm_out[i].imag = 0;
            }
        }
    }
    else
    {
        cl_out[argument.act] = 0;
        // Measured 0; now collaps this qbit.
        // XXX: this might include large errors as 
        // 1 - amplitude_1 is badly conditioned.
        partial_amplitude = 1 / sqrt(1 - amplitude_1);
        for(i = 0; i < ndim; i++)
        {
            if(i & (1 << argument.act))
            {
                qm_out[i].real = 0;
                qm_out[i].imag = 0;
            }
            else
            {
                qm_out[i].real = partial_amplitude * qm_in[i].real;
                qm_out[i].imag = partial_amplitude * qm_in[i].imag;
            }
        }
    }

    *measured_out = 1 << argument.act;
}

static char ufunc_types[5] = 
    { NPY_CDOUBLE, NPY_INT8, NPY_CDOUBLE, NPY_INT8, NPY_UINT64 };
static PyUFuncGenericFunction ufunc_X_funcs[1] = 
    { ufunc_X };
static PyUFuncGenericFunction ufunc_Z_funcs[1] = 
    { ufunc_Z };
static PyUFuncGenericFunction ufunc_H_funcs[1] = 
    { ufunc_H };
static PyUFuncGenericFunction ufunc_R_funcs[1] = 
    { ufunc_R };
static PyUFuncGenericFunction ufunc_C_funcs[1] = 
    { ufunc_C };
static PyUFuncGenericFunction ufunc_CZ_funcs[1] = 
    { ufunc_CZ };
static PyUFuncGenericFunction ufunc_M_funcs[1] = 
    { ufunc_M };


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

	if(!PyArg_ParseTuple(args, "ClldO"
                , &type
                , &(self->argument.act)
                , &(self->argument.control)
                , &(self->argument.r)
                , &(self->argument.rng))
            )
	{
		return -1;
	}

    if(!PyCallable_Check(self->argument.rng))
    {
        PyErr_SetString(PyExc_TypeError, "random (5th argument) must be a callable (returning float)");
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
		case 'Z':
		{
			self->ufunc = PyUFunc_FromFuncAndDataAndSignature(
				ufunc_Z_funcs // func
				, self->data // data
				, ufunc_types //types
				, 1 // ntypes
				, 2 // nin
				, 3 // nout
				, PyUFunc_None // identity
				, "Z_function" // name
				, "Computes the Pauli Z gate on a state." // doc
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
		case 'C':
		{
			self->ufunc = PyUFunc_FromFuncAndDataAndSignature(
				ufunc_C_funcs // func
				, self->data // data
				, ufunc_types //types
				, 1 // ntypes
				, 2 // nin
				, 3 // nout
				, PyUFunc_None // identity
				, "C_function" // name
				, "Computes the C (CNOT) gate on a state." // doc
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
		case 'M':
		{
			self->ufunc = PyUFunc_FromFuncAndDataAndSignature(
				ufunc_M_funcs // func
				, self->data // data
				, ufunc_types //types
				, 1 // ntypes
				, 2 // nin
				, 3 // nout
				, PyUFunc_None // identity
				, "M_function" // name
				, "Computes the M (Measurement) gate on a state." // doc
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
		case 'B':
		{
			self->ufunc = PyUFunc_FromFuncAndDataAndSignature(
				ufunc_CZ_funcs // func
				, self->data // data
				, ufunc_types //types
				, 1 // ntypes
				, 2 // nin
				, 3 // nout
				, PyUFunc_None // identity
				, "CZ_function" // name
				, "Computes the CZ (Controlled-Z) gate on a state." // doc
				, 0 // unused
                , "(n),(m)->(n),(m),()"); 

            if(self->ufunc == 0)
            {
                //I have no idea what is going on.
                //PyErr_SetString(PyExc_ValueError, "failed to construct the ufunc for unknow reasons");
                return -1;
            }
			break;
        }
        default:
        {
            PyErr_SetString(PyExc_ValueError, "Type must be one of X,H,R,C,M,Z,B");
            return -1;
        }
    }
    Py_INCREF(self->ufunc);
    Py_INCREF(self->argument.rng);
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

static void 
BasicGate_dealloc
    (BasicGate * self)
{
    Py_XDECREF(self->ufunc);
    Py_XDECREF(self->argument.rng);
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
	.tp_doc = "The wrapper for the basic gates X, H, Z, R_phi, CNOT, CZ and Measurement." \
              "The first argument is the type of the gate as a char: X,H,R,C,M,Z,B." 
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
    static void * basic_gates_API[basic_gates_API_pointers];
    PyObject * api_obj;

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
	if(PyModule_AddObject(module, "BasicGate", (PyObject *) &BasicGateType) < 0)
    {
        Py_XDECREF(&BasicGateType);
        Py_DECREF(module);
        return NULL;
    }

    basic_gates_API[0] = (void *)&BasicGateType;
    api_obj = PyCapsule_New((void *) basic_gates_API, "pyqcs.gates.implementations.basic_gates._C_API", NULL);
    
    if(PyModule_AddObject(module, "_C_API", api_obj) < 0)
    {
        Py_DECREF(module);
        Py_XDECREF(api_obj);
        return NULL;
    }


	return module;
}

