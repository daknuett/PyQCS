#include <iostream>
#include <dsv.hpp>

int
main(int argc, char ** argv)
{
    dsv::DSV vector(1);
    dsv::DSVOpArgument arg(0);
    std::mt19937_64 rne;
    std::vector<std::complex<double>> v1;

    vector.export_to_vector(v1);
    for(unsigned int i = 0; i < v1.size(); i++)
    {
        if(std::abs(v1[i]) > 0)
        {
            std::cout << v1[i] << "|" << i << ">\n";
        }
    }

    std::cout << "-----------------\n";
    vector.apply_op(dsv::ops::H, arg, rne);

    vector.export_to_vector(v1);
    for(unsigned int i = 0; i < v1.size(); i++)
    {
        if(std::abs(v1[i]) > 0)
        {
            std::cout << v1[i] << "|" << i << ">\n";
        }
    }
    std::cout << "-----------------\n";
    vector.apply_op(dsv::ops::H, arg, rne);
    vector.apply_op(dsv::ops::X, arg, rne);
    vector.apply_op(dsv::ops::H, arg, rne);

    vector.export_to_vector(v1);
    for(unsigned int i = 0; i < v1.size(); i++)
    {
        if(std::abs(v1[i]) > 0)
        {
            std::cout << v1[i] << "|" << i << ">\n";
        }
    }

    std::cout << "-----------------\n";
    dsv::DSV vector2 = vector;
    vector2.export_to_vector(v1);
    for(unsigned int i = 0; i < v1.size(); i++)
    {
        if(std::abs(v1[i]) > 0)
        {
            std::cout << v1[i] << "|" << i << ">\n";
        }
    }
    vector.apply_op(dsv::ops::H, arg, rne);
    vector.apply_op(dsv::ops::X, arg, rne);
    vector2.export_to_vector(v1);
    for(unsigned int i = 0; i < v1.size(); i++)
    {
        if(std::abs(v1[i]) > 0)
        {
            std::cout << v1[i] << "|" << i << ">\n";
        }
    }
    vector.export_to_vector(v1);
    for(unsigned int i = 0; i < v1.size(); i++)
    {
        if(std::abs(v1[i]) > 0)
        {
            std::cout << v1[i] << "|" << i << ">\n";
        }
    }

    std::cout << "-----------------\n";
    arg.m_phi1 = M_PI;
    vector.apply_op(dsv::ops::X, arg, rne);
    vector.apply_op(dsv::ops::R, arg, rne);

    vector.export_to_vector(v1);
    for(unsigned int i = 0; i < v1.size(); i++)
    {
        if(std::abs(v1[i]) > 0)
        {
            std::cout << v1[i] << "|" << i << ">\n";
        }
    }


    return 0;
}
