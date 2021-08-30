#include <vector>
#include <random>
#include <algorithm>
#include <rbt/rbt.hpp>

#ifndef NTEST
#define NTEST 10000
#endif

int
main(int argc, char ** argv)
{
    std::vector<int> orig;
    for(auto i = 0; i < NTEST; i++)
    {
        orig.push_back(i);
    }

    std::vector<int> copy = orig;

    std::mt19937 rne;
    std::shuffle(copy.begin(), copy.end(), rne);

    rbt::RBTree tree;
    for(auto i: copy)
    {
        tree.insert(i);
    }

    for(size_t i = 0; i < orig.size() / 2; i++)
    {
        if(!tree.has_value(i))
        {
            return -1;
        }
    }
    for(size_t i = 0; i < orig.size() / 2; i++)
    {
        if(tree.has_value(i + NTEST))
        {
            return -1;
        }
    }

    if(tree.rbt_pathlength() < 0)
    {
        return -1;
    }

    return 0;
}

