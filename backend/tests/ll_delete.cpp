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

    std::vector<int> expect;
    for(int i = 0; i < copy.size() / 2; i++)
    {
        list.delete_value(copy[i]);
        expect.push_back(copy[i + copy.size() / 2]);
    }

    std::vector<int> exported;
    list.export_vector(exported);
    std::sort(expect.begin(), expect.end());

    if(exported.size() != expect.size())
    {
        std::cerr << "exported size does not match expected size\n";
        return -1;
    }
    for(int i = 0; i < exported.size(); i++)
    {
        if(exported[i] != expect[i])
        {
            std::cerr << "mismatch at position " << i << " expect: " << expect[i] << " got: " << exported[i] << std::endl;
            return -1;
        }
    }
    return 0;
}
