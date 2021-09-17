#include <graphical2dsv.hpp>
#include <array>
#include <stdexcept>

namespace graphical2dsv
{
    const std::array<std::string, 24> C_L_decomposition_HS = 
    {
        "H", "S", "", "HS", "SH", "Z", "SHS", "ZH", "SZ", "ZHS", "HSHS", "SZHS", "HSH", "HZ", "X", "HSZ", "XS", "SHZ", "SX", "SHSZ", "ZHZ", "ZX", "HSX", "HSHSZ"
    };
    void apply_vop(dsv::DSV & vector, int qbit, uint8_t vop)
    {
        if(vop > 23)
        {
            throw std::invalid_argument("vop must be 0, ..., 23");
        }
        dsv::DSVOpArgument argument(qbit);
        for(auto c: C_L_decomposition_HS[vop])
        {
            switch(c)
            {
                case 'H':
                {
                    vector.apply_op(dsv::ops::H, argument);
                    break;
                }
                case 'S':
                {
                    vector.apply_op(dsv::ops::S, argument);
                    break;
                }
                case 'Z':
                {
                    vector.apply_op(dsv::ops::Z, argument);
                    break;
                }
                case 'X':
                {
                    vector.apply_op(dsv::ops::X, argument);
                    break;
                }
            }
        }
    }
    dsv::DSV * graphical2dsv(graphical::GraphState & state)
    {
        if(state.nqbits() > 50)
        {
            throw std::invalid_argument("conversion to dsv only allowed for at most 50 qbits");
        }
        dsv::DSV * vector = new dsv::DSV(state.nqbits());
        for(size_t i = 0; i < state.nqbits(); i++)
        {
            dsv::DSVOpArgument argument(i);
            vector->apply_op(dsv::ops::H, argument);
        }

        std::vector<int> vops;
        std::vector<std::vector<int>> ngbs;
        state.export_to_vectors(vops, ngbs);

        for(size_t i = 0; i < ngbs.size(); i++)
        {
            for(size_t j = 0; j < ngbs[i].size(); j++)
            {
                if(ngbs[i][j] >= i)
                {
                    break;
                }
                dsv::DSVOpArgument argument(i, (short int) ngbs[i][j]);
                vector->apply_op(dsv::ops::CZ, argument);
            }
        }
        for(size_t i = 0; i < vops.size(); i++)
        {
            apply_vop(*vector, i, vops[i]);
        }

        return vector;
    }

}
