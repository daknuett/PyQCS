#include <dsv.hpp>
#include <iostream>
#include "util.hpp"

int
main(int argc, char ** argv)
{
    std::mt19937_64 rne;

    dsv::DSV state(10);
    state.randomize(rne);

    if(!isclose(std::abs(state * state), 1))
    {
        std::cerr << "state is not normalized after randomize()" << std::endl;
        return -1;
    }


    dsv::DSV copy = state;

    dsv::DSVOpArgument argument(0);

    copy.apply_op(dsv::ops::X, argument);
    state.apply_op(dsv::ops::H, argument);
    state.apply_op(dsv::ops::Z, argument);
    state.apply_op(dsv::ops::H, argument);


    if(!isclose(std::abs(state * state), 1))
    {
        std::cerr << "state is not normalized after gate application" << std::endl;
        return -1;
    }
    if(!isclose(std::abs(copy * copy), 1))
    {
        std::cerr << "copy is not normalized after gate application" << std::endl;
        return -1;
    }


    if(!isclose(std::abs(state * copy), 1))
    {
        std::vector<std::complex<double>> export_state;
        std::vector<std::complex<double>> export_copy;
        state.export_to_vector(export_state);
        copy.export_to_vector(export_copy);

        std::cerr << "state:" << std::endl;
        for(auto i: export_state)
        {
            std::cerr << i << ", ";
        }
        std::cerr << std::endl;
        std::cerr << "copy:" << std::endl;
        for(auto i: export_copy)
        {
            std::cerr << i << ", ";
        }
        std::cerr << std::endl;
        return -1;
    }

    dsv::DSVOpArgument argumentC(0, (short int)1);

    copy.apply_op(dsv::ops::CX, argumentC);
    state.apply_op(dsv::ops::H, argument);
    state.apply_op(dsv::ops::CZ, argumentC);
    state.apply_op(dsv::ops::H, argument);


    if(!isclose(std::abs(state * state), 1))
    {
        std::cerr << "state is not normalized after gate application" << std::endl;
        return -1;
    }
    if(!isclose(std::abs(copy * copy), 1))
    {
        std::cerr << "copy is not normalized after gate application" << std::endl;
        return -1;
    }


    if(!isclose(std::abs(state * copy), 1))
    {
        std::vector<std::complex<double>> export_state;
        std::vector<std::complex<double>> export_copy;
        state.export_to_vector(export_state);
        copy.export_to_vector(export_copy);

        std::cerr << "state:" << std::endl;
        for(auto i: export_state)
        {
            std::cerr << i << ", ";
        }
        std::cerr << std::endl;
        std::cerr << "copy:" << std::endl;
        for(auto i: export_copy)
        {
            std::cerr << i << ", ";
        }
        std::cerr << std::endl;
        return -1;
    }

    return 0;
}
