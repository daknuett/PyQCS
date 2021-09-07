#include <rbt/rbt.hpp>
#include <time_m.hpp>
#include <iostream>


int
main(int argc, char ** argv)
{
    int const NTEST = 10;

    std::vector<int> orig;
    rbt::RBTree tree;

    for(auto i = 0; i < NTEST; i++)
    {
        orig.push_back(i);
        tree.insert(i);
    }

    std::vector<int> export_iter, export_rec;

    
    CLOG_TIMEIT_START;
    tree.export_inorder_recursive(export_rec);
    CLOG_TIMEIT_END("recursive export");
    CLOG_TIMEIT_START;
    tree.export_inorder_iterative(export_iter);
    CLOG_TIMEIT_END("iterative export");

    for(auto i: orig)
    {
        if(export_iter[i] != export_rec[i])
        {
            std::cerr << "mismatch at #" << i << std::endl;
            return -1;
        }
    }
    

    return 0;
}
