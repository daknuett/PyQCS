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

    std::vector<int> expect;
    for(int i = 0; i < copy.size() / 2; i++)
    {
        tree.delete_value(copy[i]);
        expect.push_back(copy[i + copy.size() / 2]);
    }
    if(tree.rbt_pathlength() < 0)
    {
        return -1;
    }

    std::sort(expect.begin(), expect.end());

    std::vector<int> exported;
    tree.export_inorder_recursive(exported);

    if(exported.size() != expect.size())
    {
        return -1;
    }
    for(int i = 0; i < exported.size(); i++)
    {
        if(exported[i] != expect[i])
        {
            return -1;
        }
    }
    return 0;
}

