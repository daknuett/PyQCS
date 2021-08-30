#include <ll/ll.hpp>
#include <rbt/rbt.hpp>
#include <iostream>
#include <vector>

/*
* Use this file to visually test the deletion cases.  It exports the trees
* (before and after deletion) to dot which can then be viewed and checked. This
* is very helpful for debugging.  Note that there is one case that does not get
* reproduced here (red nephew and red parent).
* 
* Use a pipeline like this:
*
*       meson compile -C builddir && builddir/main 12 true > tmp.dot && dot -Tpdf tmp.dot -o tmp.pdf && evince tmp.pdf
*       meson compile -C builddir && builddir/main 12 > tmp.dot && dot -Tpdf tmp.dot -o tmp.pdf && evince tmp.pdf
*
* This will display the tree before deletion and the tree after deletion.
* Replace evince with your PDF viewer.
* */

#define SIMPLE_DELETE_RED 1
#define SIMPLE_REPLACE_RED1 2
#define SIMPLE_REPLACE_RED2 3
#define FIXING_RED_NEPHEW1 4
#define FIXING_RED_NEPHEW2 5
#define FIXING_RED_NEPHEW3 6
#define FIXING_RED_NEPHEW4 7
#define FIXING_BLACK_NEPHEW1 8
#define FIXING_BLACK_NEPHEW2 9
#define FIXING_BLACK_NEPHEW3 10
#define FIXING_RED_SIBLING1 11
#define FIXING_RED_SIBLING2 12
#define FIXING_RED_SIBLING3 13

int
main(int argc, char ** argv)
{
    int test_case;
    if(argc < 2)
    {
         test_case = SIMPLE_DELETE_RED;
    }
    else
    {
        test_case = atoi(argv[1]);
    }
    bool show_setup = false;
    if(argc > 2)
    {
        show_setup = true;
    }

    int const to_delete = 8;

    rbt::RBTree tree;
    std::vector<int> vect;


    switch(test_case)
    {
        case SIMPLE_DELETE_RED:
        {
            tree.insert(10);
            tree.insert(12);
            tree.insert(to_delete);
            break;
        }
        case SIMPLE_REPLACE_RED1:
        {
            tree.insert(10);
            tree.insert(12);
            tree.insert(to_delete);
            tree.insert(5);
            break;
        }
        case SIMPLE_REPLACE_RED2:
        {
            tree.insert(10);
            tree.insert(12);
            tree.insert(to_delete);
            tree.insert(9);
            break;
        }
        case FIXING_RED_NEPHEW1:
        {
            tree.insert(10);
            tree.insert(to_delete);
            tree.insert(12);
            tree.insert(13);
            break;
        }
        case FIXING_RED_NEPHEW2:
        {
            tree.insert(10);
            tree.insert(to_delete);
            tree.insert(12);
            tree.insert(11);
            break;
        }
        case FIXING_RED_NEPHEW3:
        {
            tree.insert(7);
            tree.insert(to_delete);
            tree.insert(3);
            tree.insert(5);
            break;
        }
        case FIXING_RED_NEPHEW4:
        {
            tree.insert(7);
            tree.insert(to_delete);
            tree.insert(3);
            tree.insert(1);
            break;
        }
        case FIXING_BLACK_NEPHEW1:
        {
            tree.insert(10);
            tree.insert(9);
            tree.insert(to_delete);
            tree.delete_value(9);
            tree.insert(11);
            tree.insert(12);
            tree.delete_value(11);
            break;
        }
        case FIXING_BLACK_NEPHEW2:
        {
            tree.insert(10);
            tree.insert(5);
            tree.insert(11);
            tree.insert(to_delete);
            tree.insert(3);
            tree.insert(1);
            tree.insert(12);
            tree.delete_value(11);
            tree.delete_value(1);
            break;
        }
        case FIXING_BLACK_NEPHEW3:
        {
            tree.insert(30);
            tree.insert(5);
            tree.insert(50);
            tree.insert(to_delete);

            tree.insert(3);
            tree.insert(40);
            tree.insert(60);
            tree.insert(1);
            tree.delete_value(1);

            tree.insert(35);
            tree.insert(55);
            tree.insert(45);
            tree.insert(65);

            tree.insert(31);
            tree.delete_value(31);
            tree.insert(70);
            tree.delete_value(70);


            break;
        }
        case FIXING_RED_SIBLING1:
        {
            tree.insert(20);
            tree.insert(to_delete);
            tree.insert(30);
            tree.insert(25);
            tree.insert(35);
            tree.insert(31);
            tree.delete_value(31);
            break;
        }
        case FIXING_RED_SIBLING2:
        {
            tree.insert(7);
            tree.insert(to_delete);
            tree.insert(5);
            tree.insert(2);
            tree.insert(6);
            tree.insert(1);
            tree.delete_value(1);
            break;
        }
        case FIXING_RED_SIBLING3:
        {
            tree.insert(30);

        }
        default:
        {
            std::cerr << "ERR: invalid case no: " << test_case << std::endl;
            break;
        }
    }

    if(show_setup)
    {
        tree.to_dot(std::cout);
        return 0;
    }
    tree.delete_value(to_delete);
    tree.to_dot(std::cout);

    return 0;
}
