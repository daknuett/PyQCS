#include "util.hpp"

#include <graphical2dsv.hpp>

void easy_apply(dsv::DSV * vector, dsv::dsv_op op, int i)
{
    dsv::DSVOpArgument arg(i);
    vector->apply_op(op, arg);
}

void easy_apply(dsv::DSV * vector, dsv::dsv_op op, int i, short int j)
{
    dsv::DSVOpArgument arg(i, j);
    vector->apply_op(op, arg);
}

int
main(int argc, char ** argv)
{
    graphical::GraphState graph(4);
    dsv::DSV * exported;
    dsv::DSV * vector = graphical2dsv::graphical2dsv(graph);

//circuit = ((H(0) | S(1) | H(3) | S(3) | CZ(0, 2) | CZ(1, 3) | H(1) | CZ(0, 2))
//            | (X(0) | CZ(0, 1) | H(1) | S(1) | CZ(1, 2)))

    graph.apply_CL(0, 0);
    graph.apply_CL(1, 1);
    graph.apply_CL(3, 0);
    graph.apply_CL(3, 1);
    graph.apply_CZ(0, 2);
    graph.apply_CZ(1, 3);
    graph.apply_CL(1, 0);
    graph.apply_CZ(0, 2);
    graph.apply_CL(0, 14);
    graph.apply_CZ(0, 1);
    graph.apply_CL(1, 0);
    graph.apply_CL(1, 1);
    graph.apply_CZ(1, 2);

    easy_apply(vector, dsv::ops::H, 0);
    easy_apply(vector, dsv::ops::S, 1);
    easy_apply(vector, dsv::ops::H, 3);
    easy_apply(vector, dsv::ops::S, 3);
    easy_apply(vector, dsv::ops::CZ, 0, 2);
    easy_apply(vector, dsv::ops::CZ, 1, 3);
    easy_apply(vector, dsv::ops::H, 1);
    easy_apply(vector, dsv::ops::CZ, 0, 2);
    easy_apply(vector, dsv::ops::X, 0);
    easy_apply(vector, dsv::ops::CZ, 0, 1);
    easy_apply(vector, dsv::ops::H, 1);
    easy_apply(vector, dsv::ops::S, 1);
    easy_apply(vector, dsv::ops::CZ, 1, 2);


    exported = graphical2dsv::graphical2dsv(graph);

    if(!isclose(std::abs((*exported) * (*vector)), 1))
    {
        std::cerr << "expect:" << std::endl;
        vector->print_state(std::cerr);
        std::cerr << "got:" << std::endl;
        exported->print_state(std::cerr);
        std::vector<int> vops;
        std::vector<std::vector<int>> ngbs;
        graph.export_to_vectors(vops, ngbs);
        std::cerr << "vops:" << std::endl;
        for(auto v: vops)
        {
            std::cerr << v << " ";
        }
        std::cerr << std::endl;
        std::cerr << "ngbs:" << std::endl;
        for(auto n: ngbs)
        {
            std::cerr << "{";
            for(auto i: n)
            {
                std::cerr << i << ", ";
            }
            std::cerr << "}" << std::endl;
        }
        return -1;
    }

    delete exported;
    delete vector;
    return 0;
}
