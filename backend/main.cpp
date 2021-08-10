#include <ll/ll.hpp>
#include <rbt/rbt.hpp>
#include <iostream>
#include <vector>

int
main(int argc, char ** argv)
{
    ll::SortedList list;

    std::cout << "LL:\n";
    list.print();

    list.insert(10);
    list.insert(1);
    list.insert(3);
    list.print();
    list.insert(2);
    list.insert(11);
    list.insert(20);

    list.print();

    list.delete_value(3);

    list.print();

    std::cout << "has 11: " << list.has_value(11) << "\n";
    std::cout << "has 3: " << list.has_value(3) << "\n";
    std::cout << "has 21: " << list.has_value(21) << "\n";

    std::cout << "RBT:\n";

    rbt::RBTree tree;
    std::vector<int> vect;
    tree.insert(10);
    tree.insert(1);
    tree.insert(3);

    tree.export_inorder(vect);
    std::cout << "tree contains:";
    for(auto i: vect)
    {
        std::cout << ", " << i;
    }
    std::cout << "\n";
    tree.insert(2);
    tree.insert(11);
    tree.insert(20);
    tree.export_inorder(vect);
    std::cout << "tree contains:";
    for(auto i: vect)
    {
        std::cout << ", " << i;
    }
    std::cout << "\n";



    return 0;
}
