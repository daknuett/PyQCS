#ifndef generic_setup_h
#define generic_setup_h

#define PYQCS_GATE_GENERIC_SETUP \
    npy_cdouble * qm_in = (npy_cdouble *) (args[0]); \
    npy_int8 * cl_in = (npy_int8 *) (args[1]); \
    npy_cdouble * qm_out = (npy_cdouble *) (args[2]); \
    npy_int8 * cl_out = (npy_int8 *) (args[3]); \
    npy_uint64 * measured_out = (npy_uint64 *) (args[4]); \
    npy_intp qm_in_step = steps[0]; \
    npy_intp cl_in_step = steps[1]; \
    npy_intp qm_out_step = steps[2]; \
    npy_intp cl_out_step = steps[3]; \
    npy_intp ndim = dimensions[1]; \
    npy_intp nqbits = dimensions[2]

#endif
