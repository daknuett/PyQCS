#include <rbt/rbt.hpp>
#include <cstddef>
#include <stdexcept>

namespace rbt
{
    void Node::recursively_delete(void)
    {
        if(m_lower != NULL)
        {
            m_lower->recursively_delete();
            delete m_lower;
        }
        if(m_higher != NULL)
        {
            m_higher->recursively_delete();
            delete m_higher;
        }
    }

    Node::Node(Node * parent, Node * lower, Node * higher, int value):
        m_parent(parent), m_lower(lower), m_higher(higher), m_value(value)
    {
        m_color = NODE_RED;
    }

    void Node::inorder_export(std::vector<int> & vect)
    {
        if(m_lower != NULL)
        {
            m_lower->inorder_export(vect);
        }
        vect.push_back(m_value);
        if(m_higher != NULL)
        {
            m_higher->inorder_export(vect);
        }
    }
    RBTree::RBTree(void)
    {
        m_root = NULL;
    }
    RBTree::~RBTree(void)
    {
        if(m_root != NULL)
        {
            m_root->recursively_delete();
            delete m_root;
        }
    }
    Node * RBTree::do_insert(int value)
    {
        // Empty tree. Just add a (black) root node.
        if(m_root == NULL)
        {
            m_root = new Node(NULL, NULL, NULL, value);
            m_root->m_color = NODE_BLACK;
            return NULL;
        }

        // Search for a place to insert the value.
        Node * c_node = m_root;
        while(c_node != NULL)
        {
            // The value at hand is larger (i.e. higher) than
            // the current node. Go to the higher branch.
            if(c_node->m_value < value)
            {
                // It's not a leaf node. Continue searching.
                if(c_node->m_higher != NULL)
                {
                    c_node = c_node->m_higher;
                    continue;
                }
                // The higher node is a leaf. Replace it with the new node.
                Node * n_node = new Node(c_node, NULL, NULL, value);
                c_node->m_higher = n_node;
                if(c_node->m_color == NODE_RED)
                {
                    // We have to repair the RB properties.
                    return n_node;
                }
                return NULL;
            }
            // The value at hand is smaller than the current node. Go
            // to the lower branch.
            if(c_node->m_value > value)
            {
                // Continue searching.
                if(c_node->m_lower != NULL)
                {
                    c_node = c_node->m_lower;
                    continue;
                }
                // Insert the new node.
                Node * n_node = new Node(c_node, NULL, NULL, value);
                c_node->m_lower = n_node;
                if(c_node->m_color == NODE_RED)
                {
                    // We have to repair the RB properties.
                    return n_node;
                }
                return NULL;
            }
            // Should not happen in our use case.
            throw std::invalid_argument("value already in tree");
        }
        // We did not find a place to insert the value. Cannot happen.
        throw std::runtime_error("failed to insert value for unknown reasons");
    }
    inline Node * Node::get_uncle(void)
    {
        if(this->m_parent->m_parent->m_lower != this->m_parent)
        {
            return this->m_parent->m_parent->m_lower;
        }
        return this->m_parent->m_parent->m_higher;
    }
    inline bool Node::is_lower_child(void)
    {
        if(this->m_parent == NULL)
        {
            return false;
        }
        if(this->m_parent->m_lower == this)
        {
            return true;
        }
        return false;
    }
    void RBTree::repair(Node * causing_node)
    {
        // Root node. Repair procedure is to set it to black.
        if(causing_node->m_parent == NULL)
        {
            causing_node->m_color = NODE_BLACK;
            return;
        }

        // We are not at the root node.

        // Check if we are done. I think this can happen if we entered the
        // recursion below.
        if(causing_node->m_parent->m_color == NODE_BLACK)
        {
            return;
        }
        
        // Parent is also red. This guarantees that we have
        // an uncle since the parent cannot be the root.
        Node * uncle = causing_node->get_uncle();
        
        // Red uncle. Set the Parent and the uncle to black
        // and the grandparent to red. Then repair the grandparent.
        if(uncle != NULL && uncle->m_color == NODE_RED)
        {
            uncle->m_color = NODE_BLACK;
            causing_node->m_parent->m_color = NODE_BLACK;
            causing_node->m_parent->m_parent->m_color = NODE_RED;
            repair(causing_node->m_parent->m_parent);
            return;
        }

        // Black uncle. Rotate the tree the one way or the other.
        if(causing_node->m_parent->is_lower_child())
        {
            if(!causing_node->is_lower_child())
            {
                // We have a tree like this:
                //
                //           C            |
                //      (c)>/ \           |
                //         /   \          |
                //        A    delta      |
                //       / \              |
                //      /   \<(b)         |
                //     /     \            |
                //  alpha     B           |
                //           / \          |
                //      (a)>/   \         |
                //         /     \        |
                //      beta    gamma     |
                //
                // Make the node B (causing_node) a lower child:
                //
                //              C         |
                //             / \        | 
                //        (c)>/   \       | 
                //           /     \      | 
                //          B     delta   | 
                //         / \            | 
                //    (b)>/   \           | 
                //       A   gamma        | 
                //      / \               | 
                //     /   \<(a)          | 
                //    /     \             | 
                //  alpha   beta          | 
                //      
                Node * A = causing_node->m_parent; // A
                Node * beta = causing_node->m_lower; // beta
                Node * C = A->m_parent; // C
                Node * B = causing_node;

                C->m_lower = B; // (c)
                B->m_parent = C; // (c)
                B->m_lower = A; // (b)
                A->m_parent = B; // (b)
                A->m_higher = beta; // (a)
                if(beta != NULL)
                {
                    beta->m_parent = A; // (a)
                }

                causing_node = A; // Now repair around A which was prevously the parent.
            }
            // We have a tree like this:
            //
            //          (c)>|         |
            //              C         |
            //             / \        | 
            //        (b)>/   \       | 
            //           /     \      | 
            //          B     delta   | 
            //         / \            | 
            //        /   \<(a)       | 
            //       A   gamma        | 
            //      / \               | 
            //     /   \              | 
            //    /     \             | 
            //  alpha   beta          | 
            //      
            // Right-rotate around the parent to get to this tree:
            //
            //         |<(c)          |
            //         B              |
            //        / \             | 
            //       /   \<(b)        | 
            //      /     \           | 
            //     A       C          | 
            //    / \ (a)>/ \         | 
            //   /   \   /   \        | 
            // alpha | gamma |        | 
            //     beta    delta      | 
            //
            Node * B = causing_node->m_parent; // B
            Node * C = B->m_parent; // C
            Node * greatgrandparent = C->m_parent; // C's parent
            Node * gamma = B->m_higher; // B's right child

            C->m_lower = gamma; // (a)
            if(gamma != NULL)
            {
                gamma->m_parent = C; // (a)
            }
            B->m_higher = C; // (b)
            C->m_parent = B; // (b)
            if(greatgrandparent == NULL)
            {
                m_root = B; // (c)
            }
            else
            {
                if(C->is_lower_child())
                {
                    greatgrandparent->m_lower = B; // (c)
                }
                else
                {
                    greatgrandparent->m_higher = B; // (c)
                }
            }
            B->m_parent = greatgrandparent; //(c)

            B->m_color = NODE_BLACK;
            C->m_color = NODE_RED;
            return;
        }

        if(causing_node->is_lower_child())
        {
            // We have a tree like this:
            //                                  
            //                                  |
            //              C                   |
            //             / \                  |
            //            /   \<(c)             |
            //           /     \                |
            //        delta     A               |
            //                 / \              |
            //            (b)>/   \             |
            //               /     \            |
            //              B    gamma          |
            //             / \                  |
            //            /   \<(a)             |
            //           /     \                |
            //        alpha   beta              |
            //                                  |
            //
            // And we want to make B (causing_node) a right child:
            //
            //                                  |
            //              C                   |
            //             / \                  |
            //            /   \<(b)             |
            //           /     \                |
            //        delta     B               |
            //                 / \              |
            //                /   \<(c)         |
            //               /     \            |
            //           alpha      A           |
            //                     / \          |
            //                (a)>/   \         |
            //                   /     \        |
            //                beta    gamma     |
            //                                  |
            //
            Node * A = causing_node->m_parent; // A
            Node * beta = causing_node->m_higher; // beta
            Node * C = A->m_parent; // C
            Node * B = causing_node;

            C->m_higher = B; // (b)
            B->m_parent = C; // (b)
            B->m_higher = A; // (c)
            A->m_parent = B; // (c)
            A->m_lower = beta; // (a)
            if(beta != NULL)
            {
                beta->m_parent = A; // (a)
            }

            causing_node = A; // Now repair around A which was previously the parent.
        }
        
        // We have a tree like this (A is the causing node)
        //
        //                                  |
        //              |<(c)               |
        //              C                   |
        //             / \                  |
        //            /   \<(b)             |
        //           /     \                |
        //        delta     B               |
        //                 / \              |
        //            (a)>/   \             |
        //               /     \            |
        //           gamma      A           |
        //                     / \          |
        //                    /   \         |
        //                   /     \        |
        //                alpha   beta      |
        //                                  |
        //
        // And we want to left-rotate it to get a flatter tree:
        //
        //                                  |
        //               |<(c)              |
        //               B                  |
        //              / \                 |
        //             /   \                |
        //        (b)>/     \               |
        //           /       \              |
        //          /         \             |
        //         C           A            |
        //        / \         / \           |
        //       /   \<(a)   /   \          |
        //      /     \     /     \         |
        //   delta    |  alpha   beta       |
        //         gamma                    |
        //                                  |
        //                                  |
        Node * B = causing_node->m_parent; // B
        Node * C = B->m_parent; // C
        Node * greatgrandparent = C->m_parent; // C's B
        Node * gamma = B->m_lower; // B's left child

        C->m_higher = gamma; // (a)
        if(gamma != NULL)
        {
            gamma->m_parent = C; // (a)
        }
        B->m_lower = C; // (b)
        C->m_parent = B; // (b)
        if(greatgrandparent == NULL)
        {
            m_root = B; // (c)
        }
        else
        {
            if(C->is_lower_child())
            {
                greatgrandparent->m_lower = B; // (c)
            }
            else
            {
                greatgrandparent->m_higher = B; // (c)
            }
        }
        B->m_parent = greatgrandparent; // (c)

        C->m_color = NODE_RED;
        B->m_color = NODE_BLACK;
        
    }
    void RBTree::insert(int value)
    {
        Node * n_node = do_insert(value);
        // enable this line if you do not want a red black tree.
        //return;
        if(n_node != NULL)
        {
            repair(n_node);
        }
    }
    void RBTree::export_inorder(std::vector<int> & vect)
    {
        vect.resize(0);
        if(m_root != NULL)
        {
            m_root->inorder_export(vect);
        }
    }
}
