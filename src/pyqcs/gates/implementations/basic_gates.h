#ifndef basic_gates_h
#define basic_gates_h

#include <Python.h>

#define basic_gates_API_pointers 1


#ifdef basic_gates_module

#else // basic_gates_module

static void ** basic_gates_API;

#define BasicGateType \
    (static PyTypeObject *) basic_gates_API[0]

static int
import_basic_gates(void)
{
    basic_gates_API = (void **) PyCapsule_Import("pyqcs.gates.implementations.basic_gates._C_API", 0);
    return (basic_gates_API != NULL) ? 0: -1;
}

#endif // basic_gates_module

#endif
