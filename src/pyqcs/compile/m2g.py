"""
This module is supposed to compile matrices (numpy NxN arrays) to 
ufuncs that can be applied to some naive states.
"""

from numpy import ndarray, identity, isclose
from numpy.linalg import det

class M2GCompiler(object):
    def __init__(self):
        self._c2s_backends = {2: self._do_c2s_2}
        self.code_stump = \
        '''
        static void
        ufunc_compled_gate( char ** args
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
                %s
            }
            *measured_out = 0;
        }
        '''

    def compile_to_str(self, m):
        """
        Does some type checking and then compiles the matrix m
        to numpy ufunc generic function source code.
        """
        if(not isinstance(m, ndarray)):
            raise TypeError("matrices must be numpy ndarrays")
        if(len(m.shape) != 2):
            raise ValueError("matrices must be of dimension NxN")
        if(m.shape[0] != m.shape[1]):
            raise ValueError("matrices must be of dimension NxN")

        test = m.dot(m.conj().T)
        if(not isclose(test, identity(m.shape[0])).all()):
            raise ValueError("matrices must be unitary")

        if(not isclose(det(m), 1).all()):
            raise ValueError("matrices must be in SU(N)")

        if(m.shape[0] not in self._c2s_backends):
            raise NotImplementedError("compiling {0}x{0} arrays is not implemented".format(m.shape[0]))

        return self._c2s_backends[m.shape[0]](m)

    def _do_c2s_2(self, m):
        # The following naming convention is used:
        # up === u := |1>
        # down === d := |0>
        #
        # This yields the following matrix form:
        #
        # (u2u, d2u)
        # (u2d, d2d)
        #
        # Where a2b means a mapping from state a to state b.
        
        u2u = m[0,0]
        d2d = m[1,1]
        u2d = m[1,0]
        d2u = m[0,1]

        
        code_fragment = \
        '''
        npy_intp not_index = i ^ (1 << argument.act);

        if(!(i & (1 << argument.act)))
        {
            %s
        }
        else
        {
            %s
        }
        '''

        computation = \
        '''
        qm_out[i].real =  qm_in[i].real * {0.real} + qm_in[not_index].real * {1.real} - qm_in[i].imag * {0.imag} - qm_in[not_index].imag * {1.imag};
        qm_out[i].imag =  qm_in[i].imag * {0.real} + qm_in[not_index].imag * {0.real} + qm_in[i].real * {0.imag} + qm_in[not_index].real * {1.imag};
        '''

        loop_code = code_fragment % (computation.format(u2u, d20) , computation.format(d2d, u2d))



        return self.code_stump % loop_code


