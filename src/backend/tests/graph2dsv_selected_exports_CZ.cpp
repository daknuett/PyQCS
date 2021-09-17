#include "util.hpp"

#include <graphical2dsv.hpp>

int
main(int argc, char ** argv)
{
    graphical::GraphState graph(3);
    dsv::DSV * exported;
    dsv::DSV * vector = graphical2dsv::graphical2dsv(graph);

    dsv::DSVOpArgument arg(0, (short int)1);
    graph.apply_CZ(0, 1);
    vector->apply_op(dsv::ops::CZ, arg);
    
    
    exported = graphical2dsv::graphical2dsv(graph);
    if(!isclose(std::abs((*vector) * (*exported)), 1))
    {
        std::cerr << "FAILED after CZ(0, 1)" << std::endl;
        delete exported;
        return -1;
    }
    delete exported;

    arg = dsv::DSVOpArgument(0, (short int) 2);
    graph.apply_CZ(0, 2);
    vector->apply_op(dsv::ops::CZ, arg);

    exported = graphical2dsv::graphical2dsv(graph);
    if(!isclose(std::abs((*vector) * (*exported)), 1))
    {
        std::cerr << "FAILED after CZ(0, 2)" << std::endl;
        std::cerr << "expect:" << std::endl;
        vector->print_state(std::cerr);
        std::cerr << "got:" << std::endl;
        exported->print_state(std::cerr);
        std::vector<int> vops;
        std::vector<std::vector<int>> ngbs;
        graph.export_to_vectors(vops, ngbs);
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
        delete exported;
        return -1;
    }
    delete exported;

    arg = dsv::DSVOpArgument(1, (short int) 2);
    graph.apply_CZ(1, 2);
    vector->apply_op(dsv::ops::CZ, arg);

    exported = graphical2dsv::graphical2dsv(graph);
    if(!isclose(std::abs((*vector) * (*exported)), 1))
    {
        std::cerr << "FAILED after CZ(1, 2)" << std::endl;
        std::cerr << "expect:" << std::endl;
        vector->print_state(std::cerr);
        std::cerr << "got:" << std::endl;
        exported->print_state(std::cerr);
        std::vector<int> vops;
        std::vector<std::vector<int>> ngbs;
        graph.export_to_vectors(vops, ngbs);
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
        delete exported;
        return -1;
    }
    delete exported;

    arg = dsv::DSVOpArgument(0);
    graph.apply_CL(0, 0);
    vector->apply_op(dsv::ops::H, arg);
    arg = dsv::DSVOpArgument(0, (short int) 1);
    graph.apply_CZ(0, 1);
    vector->apply_op(dsv::ops::CZ, arg);


    exported = graphical2dsv::graphical2dsv(graph);
    if(!isclose(std::abs((*vector) * (*exported)), 1))
    {
        std::cerr << "FAILED after H(0) | CZ(0, 1)" << std::endl;
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
        delete exported;
        return -1;
    }
    delete exported;



    delete vector;
    return 0;
}
