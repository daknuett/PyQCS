#pragma once
#include <complex>
#include <random>


namespace dsv
{
    class DSVOpArgument
    {
        public:
        unsigned short int m_act;
        short int m_control;
        double m_phi1;
        double m_phi2;
        double m_phi3;
        DSVOpArgument(unsigned short int act);
        DSVOpArgument(unsigned short int act, short int control);
        DSVOpArgument(unsigned short int act, double phi);
    };

    typedef int (* dsv_op)(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument);

    class DSV
    {
        private:
        unsigned int m_ndims;
        unsigned short int m_nqbits;
        unsigned int m_cvect;
        std::complex<double> *m_vect[2];
        public:
        unsigned short int nqbits(void);
        DSV(unsigned short int nqbits);
        DSV(const DSV & copy);
        ~DSV(void);
        int apply_op(dsv_op op, DSVOpArgument & argument);
        /**
         * Exports the current dsv data to ``vector''. This makes a copy.
         * */
        void export_to_vector(std::vector<std::complex<double>> & vector);
        /**
         * Computes the overlap of ``this'' and ``other''.
         * */
        std::complex<double> operator*(DSV & other);
        /**
         * Randomizes the state where each amplitude is drawn from a uniform
         * distribution.
         * */
        void randomize(std::mt19937_64 & rne);
        void print_state(std::ostream & output);
        /**
         * Explicitly normalize the state to 1. This can be used when some
         * operation might have broken the normalization (for instance
         * randomization or execution of many gates).
         * */
        void normalize(void);
        /**
         * Compute the probability to measure ``|1>'' on qbit ``i''. 
         * (i.e. the overlap squared of ``|(1 << i)>''). Result is between
         * 0 and 1.
         * */
        double measurement_probability(unsigned short int i);
        /**
         * Projects the ``i'' th qbit to ``value'' (= 0, 1) in computational
         * (Pauli Z) basis. The resulting state is normalized to 1.
         *
         * FIXME: currently the implementation has a bug where the resulting
         * states are not always normalized. We fixed this by calling
         * ``normalize'' but this should be avoided for performance reasons.
         *
         * */
        void project_to(unsigned short int i, int value);
        /**
         * This method exports two vectors (of same length) of the possible
         * measuring outcomes and the respective probabilities. Only outcomes
         * that have a likelyhood above ``eps'' are included.
         *
         * The outcomes are stored in ``labels'' and are the bitstrings. The
         * probabilities are stored in ``probabilities''.
         * */
        void statistic(std::vector<unsigned int> & labels, std::vector<double> & probabilities, double eps);
        /**
         * This method exports the vector to a C array.
         * We need this for inter-operability with numpy (to allow
         * numpy to free the memory when an ndarray is deleted).
         *
         * One should not use this for anything else, use the ``export_to_vector''
         * method instead.
         *
         * Returns the length of ``*array''.
         * */
        size_t export_to_array(std::complex<double> ** array);

    };


    namespace ops
    {
        int CX(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument);
        int CZ(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument);
        int Z(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument);
        int X(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument);
        int S(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument);
        int H(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument);
        int R(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument);

    }
}
