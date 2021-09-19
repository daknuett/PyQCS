#include <graphical.hpp>

int
main(int argc, char ** argv)
{
    graphical::GraphState * state1 = new graphical::GraphState(5);
    state1->apply_CZ(0, 2);
    state1->apply_CZ(0, 3);

    graphical::GraphState * state2 = new graphical::GraphState(*state1);

    delete state1;
    delete state2;
    return 0;
}
