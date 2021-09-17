#pragma once

#include <dsv.hpp>
#include <graphical.hpp>

namespace graphical2dsv
{
    void apply_vop(dsv::DSV & vector, int qbit, uint8_t vop);
    dsv::DSV * graphical2dsv(graphical::GraphState & state);
}
