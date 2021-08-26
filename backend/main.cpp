#include <ll/ll.hpp>
#include <rbt/rbt.hpp>
#include <iostream>
#include <vector>

int
main(int argc, char ** argv)
{

    rbt::RBTree tree;
    std::vector<int> vect;

    tree.insert(5);
    tree.insert(4);
    tree.insert(3);
    tree.insert(2);
    tree.insert(1);

    tree.insert(11);
    tree.insert(20);
    tree.delete_value(2);
    tree.delete_value(1);
    tree.delete_value(3);
    tree.delete_value(20);
    tree.delete_value(5);
    tree.to_dot(std::cout);
    return 0;
}
