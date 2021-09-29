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
        void export_to_vector(std::vector<std::complex<double>> & vector);
        std::complex<double> operator*(DSV & other);
        void randomize(std::mt19937_64 & rne);
        void print_state(std::ostream & output);
        void normalize(void);
        double measurement_probability(unsigned short int i);
        void project_to(unsigned short int i, int value);
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
