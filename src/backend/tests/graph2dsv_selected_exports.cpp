#include "util.hpp"

#include <graphical2dsv.hpp>

int
main(int argc, char ** argv)
{
    dsv::DSV vector(1);
    graphical::GraphState graph(1);

    dsv::DSVOpArgument arg(0);
    vector.apply_op(dsv::ops::H, arg);

    dsv::DSV * exported = graphical2dsv::graphical2dsv(graph);

    if(!isclose(std::abs((*exported) * (*exported)), 1))
    {
        std::cerr << "FAILED: no VOP applied" << std::endl;
        std::cerr << "unitarity check failed" << std::endl;
        delete exported;
        return -1;
    }
    if(!isclose(std::abs(vector * (*exported)), 1))
    {
        std::cerr << "FAILED: no VOP applied" << std::endl;
        delete exported;
        return -1;
    }
    delete exported;

    graph.apply_CL(0, 0);
    vector.apply_op(dsv::ops::H, arg);
    exported = graphical2dsv::graphical2dsv(graph);

    if(!isclose(std::abs((*exported) * (*exported)), 1))
    {
        std::cerr << "FAILED: H applied: unitarity check failed" << std::endl;
        delete exported;
        return -1;
    }
    if(!isclose(std::abs(vector * (*exported)), 1))
    {
        std::cerr << "FAILED: H applied" << std::endl;
        delete exported;
        return -1;
    }
    delete exported;

    graph.apply_CL(0, 14);
    vector.apply_op(dsv::ops::X, arg);
    exported = graphical2dsv::graphical2dsv(graph);

    if(!isclose(std::abs((*exported) * (*exported)), 1))
    {
        std::cerr << "FAILED: X applied" << std::endl;
        std::cerr << "unitarity check failed" << std::endl;
        delete exported;
        return -1;
    }

    if(!isclose(std::abs(vector * (*exported)), 1))
    {
        std::cerr << "FAILED: X applied" << std::endl;
        std::cerr << "expect:" << std::endl;
        vector.print_state(std::cerr);
        std::cerr << "got:" << std::endl;
        exported->print_state(std::cerr);

        delete exported;
        return -1;
    }
    delete exported;


    return 0;
}
