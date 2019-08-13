#ifndef generic_setup_h
#define generic_setup_h

#define PYQCS_GATE_GENERIC_SETUP \
    npy_cdouble * qm_in = args[0]; \
    npy_double * cl_in = args[1]; \
    npy_cdouble * qm_out = args[2]; \
    npy_double * cl_out = args[3]; \
    npy_unint64 * measured_out = args[4]; \
    npy_intp * qm_in_step = steps[0]; \
    npy_intp * cl_in_step = steps[1]; \
    npy_intp * qm_out_step = steps[2]; \
    npy_intp * cl_out_step = steps[3]; \
    npy_intp * ndim = dimensions[1]; \
    npy_intp * nqbits = dimensions[2]
    
    


#endif
