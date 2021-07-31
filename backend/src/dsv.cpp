#include <dsv.hpp>
#include <iostream>

namespace dsv
{
    DSV::DSV(unsigned short int nqbits)
    {
        if(!nqbits)
        {
            throw std::invalid_argument("nqbits must be > 0");
        }
        if(nqbits > 30)
        {
            throw std::invalid_argument("nqbits must be <= 30");
        }
        m_nqbits = nqbits;
        m_ndims = 1 << nqbits;
        m_cvect = 0;

        m_vect[0] = new std::complex<double>[m_ndims]();
        m_vect[1] = new std::complex<double>[m_ndims]();

        m_vect[0][0] = 1;
    }
    DSV::DSV(const DSV & copy)
    {
        m_nqbits = copy.m_nqbits;
        m_ndims = copy.m_ndims;
        m_cvect = copy.m_cvect;

        m_vect[0] = new std::complex<double>[m_ndims]();
        m_vect[1] = new std::complex<double>[m_ndims]();

        std::copy(copy.m_vect[copy.m_cvect], copy.m_vect[copy.m_cvect] + copy.m_ndims, m_vect[m_cvect]);
    }
    DSV::~DSV(void)
    {
        delete [] m_vect[0];
        delete [] m_vect[1];
    }


    int DSV::apply_op(dsv_op op, DSVOpArgument & argument, std::mt19937_64 & rne)
    {
        if(argument.m_act > m_nqbits)
        {
            throw std::invalid_argument("act > nqbits");
        }
        if(argument.m_control >= 0 && argument.m_act == argument.m_control)
        {
            throw std::invalid_argument("act = control");
        }
        if(argument.m_control >= 0 && argument.m_control > m_nqbits)
        {
            throw std::invalid_argument("control > nqbits");
        }
        m_cvect ^= 1;
        return op(m_vect[m_cvect^1], m_vect[m_cvect], m_ndims, argument, rne);
    }

    void DSV::export_to_vector(std::vector<std::complex<double>> & vector)
    {
        vector.resize(0);
        vector.reserve(m_ndims);
        for(unsigned int i = 0; i < m_ndims; i++)
        {
            vector.push_back(m_vect[m_cvect][i]);
        }
    }

    DSVOpArgument::DSVOpArgument(unsigned short int act)
    {
        m_act = act;
        m_control = -1;
        m_phi1 = 0;
        m_phi2 = 0;
        m_phi3 = 0;
    }
    DSVOpArgument::DSVOpArgument(unsigned short int act, short int control)
    {
        m_act = act;
        m_control = control;
        m_phi1 = 0;
        m_phi2 = 0;
        m_phi3 = 0;
    }
    DSVOpArgument::DSVOpArgument(unsigned short int act, double phi)
    {
        m_act = act;
        m_control = -1;
        m_phi1 = phi;
        m_phi2 = 0;
        m_phi3 = 0;
    }


    namespace ops
    {
        int CX(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument, std::mt19937_64 & rne)
        {
            for(unsigned int i = 0; i < ndims; i++)
            {
                if(i & (1 << argument.m_control))
                {
                    out[i] = in[i ^ (1 << argument.m_act)];
                }
                else
                {
                    out[i] = in[i];
                }
            }
            return 0;
        }
        int CZ(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument, std::mt19937_64 & rne)
        {
            for(unsigned int i = 0; i < ndims; i++)
            {
                if(i & (1 << argument.m_control) && i & (1 << argument.m_act))
                {
                    out[i] = -in[i];
                }
                else
                {
                    out[i] = in[i];
                }
            }
            return 0;
        }
        int Z(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument, std::mt19937_64 & rne)
        {
            for(unsigned int i = 0; i < ndims; i++)
            {
                if(i & (1 << argument.m_act))
                {
                    out[i] = -in[i];
                }
                else
                {
                    out[i] = in[i];
                }
            }
            return 0;
        }
        int X(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument, std::mt19937_64 & rne)
        {
            for(unsigned int i = 0; i < ndims; i++)
            {
                out[i] = in[i ^ (1 << argument.m_act)];
            }
            return 0;
        }
        int H(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument, std::mt19937_64 & rne)
        {
            for(unsigned int i = 0; i < ndims; i++)
            {
                if(i & (1 << argument.m_act))
                {
                    out[i] = (in[i ^ (1 << argument.m_act)] - in[i]) * M_SQRT1_2;
                }
                else
                {
                    out[i] = (in[i ^ (1 << argument.m_act)] + in[i]) * M_SQRT1_2;
                }

            }
            return 0;
        }
        int R(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument, std::mt19937_64 & rne)
        {
            std::complex<double> i(0., 1.);
            std::complex<double> c = std::exp(1.*i * argument.m_phi1);
            for(unsigned int i = 0; i < ndims; i++)
            {
                if(i & (1 << argument.m_act))
                {
                    out[i] = c*in[i];
                }
                else
                {
                    out[i] = in[i];
                }
            }
            return 0;
        }

    }
}
