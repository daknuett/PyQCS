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

    typedef int (* dsv_op)(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument, std::mt19937_64 & rne);

    class DSV
    {
        private:
        unsigned int m_ndims;
        unsigned short int m_nqbits;
        unsigned int m_cvect;
        std::complex<double> *m_vect[2];
        public:
        DSV(unsigned short int nqbits);
        DSV(const DSV & copy);
        ~DSV(void);
        int apply_op(dsv_op op, DSVOpArgument & argument, std::mt19937_64 & rne);
        void export_to_vector(std::vector<std::complex<double>> & vector);
    };


    namespace ops
    {
        int CX(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument, std::mt19937_64 & rne);
        int CZ(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument, std::mt19937_64 & rne);
        int Z(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument, std::mt19937_64 & rne);
        int X(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument, std::mt19937_64 & rne);
        int H(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument, std::mt19937_64 & rne);
        int R(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument, std::mt19937_64 & rne);

    }
}
