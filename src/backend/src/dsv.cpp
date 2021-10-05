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

    unsigned short int DSV::nqbits(void)
    {
        return m_nqbits;
    }

    int DSV::apply_op(dsv_op op, DSVOpArgument & argument)
    {
        if(argument.m_act >= m_nqbits)
        {
            throw std::invalid_argument("act > nqbits");
        }
        if(argument.m_control >= 0 && argument.m_act == argument.m_control)
        {
            throw std::invalid_argument("act = control");
        }
        if(argument.m_control >= 0 && argument.m_control >= m_nqbits)
        {
            throw std::invalid_argument("control > nqbits");
        }
        m_cvect ^= 1;
        return op(m_vect[m_cvect^1], m_vect[m_cvect], m_ndims, argument);
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

    std::complex<double> DSV::operator*(DSV & other)
    {
        if(other.m_ndims != m_ndims)
        {
            throw std::invalid_argument("DSVs must have same dimension for operator *");
        }
        std::complex<double> result(0, 0);

        for(size_t i = 0; i < m_ndims; i++)
        {
            result += m_vect[m_cvect][i] * std::conj(other.m_vect[other.m_cvect][i]);
        }

        return result;
    }
    void DSV::randomize(std::mt19937_64 & rne)
    {
        std::uniform_real_distribution<double> reals(0, 1);
        std::uniform_real_distribution<double> imags(0, 1);

        for(size_t i = 0; i < m_ndims; i++)
        {
            std::complex<double> cmplx(reals(rne), imags(rne));
            m_vect[m_cvect][i] = cmplx;
        }

        std::complex<double> overlap = (*this)*(*this);
        double scale = std::sqrt(std::abs(overlap));

        for(size_t i = 0; i < m_ndims; i++)
        {
            m_vect[m_cvect][i] /= scale;
        }
    }

    void DSV::normalize(void)
    {
        double amplitude = std::abs((*this) * (*this));
        double normalization = std::sqrt(amplitude);

        for(size_t i = 0; i < m_ndims; i++)
        {
            m_vect[m_cvect][i] /= normalization;
        }
    }
    double DSV::measurement_probability(unsigned short int i)
    {
        if(i >= m_nqbits)
        {
            throw std::invalid_argument("qbit out of range");
        }

        double amplitude_1 = 0, amplitude_0 = 0;
        for(size_t j = 0; j < m_ndims; j++)
        {
            if(j & (1 << i))
            {
                amplitude_1 += std::abs(m_vect[m_cvect][j]);
            }
            else
            {
                amplitude_0 += std::abs(m_vect[m_cvect][j]);
            }
        }

        // Try to account for numerical errors that happen when adding small and
        // large elements.
        amplitude_1 -= (amplitude_1 + amplitude_0) / 2 - 0.5;
    
        return amplitude_1;
    }

    void DSV::project_to(unsigned short int i, int value)
    {
        double amplitude_1 = measurement_probability(i);
        if(value)
        {
            double normalization = std::sqrt(amplitude_1);
            for(size_t j = 0; j < m_ndims; j++)
            {
                if(j & (1 << i))
                {
                    m_vect[m_cvect^1][j] = m_vect[m_cvect][j] / normalization;
                }
                else
                {
                    m_vect[m_cvect^1][j] = 0;
                }
            }
        }
        else
        {
            double normalization = std::sqrt(1 - amplitude_1);
            for(size_t j = 0; j < m_ndims; j++)
            {
                if(j & (1 << i))
                {
                    m_vect[m_cvect^1][j] = 0;
                }
                else
                {
                    m_vect[m_cvect^1][j] = m_vect[m_cvect][j] / normalization;
                }
            }
        }

        m_cvect ^= 1;
        // FIXME: XXX:
        // This should not be necessary. The routine above should give
        // a normalized state but for some reason there are cases where the
        // resulting state is not normalized.  We should fix the routine above
        // and remove the extra normalization.
        normalize();
    }
    void DSV::statistic(std::vector<unsigned int> & labels, std::vector<double> & probabilities, double eps)
    {
        labels.resize(0);
        probabilities.resize(0);

        for(size_t i = 0; i < m_ndims; i++)
        {
            if(std::abs(m_vect[m_cvect][i]) > eps)
            {
                labels.push_back(i);
                probabilities.push_back(std::abs(m_vect[m_cvect][i]));
            }
        }
    }

    void DSV::print_state(std::ostream & output)
    {
        double const eps = 1e-4;

        for(int i = 0; i < m_ndims; i++)
        {
            if(std::abs(m_vect[m_cvect][i]) > eps)
            {
                output << "\t+" << m_vect[m_cvect][i] << " |" << i << ">" << std::endl;
            }
        }
    }

    size_t DSV::export_to_array(std::complex<double> ** array)
    {
        *array = new std::complex<double>[m_ndims];

        for(size_t i = 0; i < m_ndims; i++)
        {
            (*array)[i] = m_vect[m_cvect][i];
        }
        return m_ndims;
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
        int CX(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument)
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
        int CZ(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument)
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
        int Z(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument)
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
        int S(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument)
        {
            std::complex<double> j(0., 1.);
            for(unsigned int i = 0; i < ndims; i++)
            {
                if(i & (1 << argument.m_act))
                {
                    out[i] = j*in[i];
                }
                else
                {
                    out[i] = in[i];
                }
            }
            return 0;
        }
        int X(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument)
        {
            for(unsigned int i = 0; i < ndims; i++)
            {
                out[i] = in[i ^ (1 << argument.m_act)];
            }
            return 0;
        }
        int H(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument)
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
        int R(std::complex<double> * in, std::complex<double> * out, unsigned int ndims, DSVOpArgument & argument)
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
