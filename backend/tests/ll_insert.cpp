#include <vector>
#include <random>
#include <algorithm>
#include <ll/ll.hpp>

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

    ll::SortedList list;
    for(auto i: copy)
    {
        list.insert(i);
    }

    std::vector<int> exported;
    list.export_vector(exported);

    if(exported.size() != orig.size())
    {
        return -1;
    }
    for(int i = 0; i < exported.size(); i++)
    {
        if(exported[i] != orig[i])
        {
            return -1;
        }
    }
    return 0;
}
