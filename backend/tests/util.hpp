#pragma once


#include <cmath>
#include <iostream>
#ifndef EPSILON
#define EPSILON 1e-8
#endif

bool isclose(double a, double b)
{
    if(std::abs(a - b) >= EPSILON)
    {
        std::cerr << "isclose: " << a << " != " << b << " +/- " << EPSILON << std::endl;
    }
    return std::abs(a - b) < EPSILON;
}
