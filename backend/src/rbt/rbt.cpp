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

    Node::Node(Node * parent, int value, char marker):
        m_parent(parent), m_value(value), m_marker(marker)
    {
        m_color = NODE_RED;
        m_higher = m_lower = NULL;
    }

    void Node::recursive_inorder_export(std::vector<int> & vect)
    {
        if(m_lower != NULL)
        {
            m_lower->recursive_inorder_export(vect);
        }
        vect.push_back(m_value);
        if(m_higher != NULL)
        {
            m_higher->recursive_inorder_export(vect);
        }
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


    void Node::repair_markers(char marker)
    {
        if(m_lower != NULL)
        {
            m_lower->repair_markers(marker);
        }
        m_marker = marker;
        if(m_higher != NULL)
        {
            m_higher->repair_markers(marker);
        }
    }
    RBTree::RBTree(void)
    {
        m_root = NULL;
        m_marker_mask = 0;
        m_marker_sanity = 1;
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
            m_root = new Node(NULL, value, m_marker_mask);
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
                Node * n_node = new Node(c_node, value, m_marker_mask);
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
                Node * n_node = new Node(c_node, value, m_marker_mask);
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

    inline void RBTree::repair_markers_if_needed(void)
    {
        if(!m_marker_sanity)
        {
            if(m_root == NULL)
            {
                m_marker_sanity = 1;
                return;
            }
            m_root->repair_markers(m_marker_mask);
            m_marker_sanity = 1;
        }
    }

    void RBTree::repair_after_insert(Node * causing_node)
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
            repair_after_insert(causing_node->m_parent->m_parent);
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

            causing_node->m_parent->m_color = NODE_BLACK;
            causing_node->m_parent->m_parent->m_color = NODE_RED;
            right_rotate(causing_node->m_parent); // Look at this function to see what is going on.

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
        
        causing_node->m_parent->m_parent->m_color = NODE_RED;
        causing_node->m_parent->m_color = NODE_BLACK;
        left_rotate(causing_node->m_parent); // Look at this function to see what is going on.
    }
    void RBTree::insert(int value)
    {
        Node * n_node = do_insert(value);
        // enable this line if you do not want a red black tree.
        //return;
        if(n_node != NULL)
        {
            repair_after_insert(n_node);
        }
    }
    void RBTree::export_inorder_recursive(std::vector<int> & vect)
    {
        vect.resize(0);
        if(m_root != NULL)
        {
            m_root->recursive_inorder_export(vect);
        }
    }
    void Node::dot_edges(std::ostream & stream)
    {
        if(m_lower != NULL)
        {
            m_lower->dot_edges(stream);
            stream << "\t" << m_value << " -> " << m_lower->m_value << "\n";
        }
        if(m_higher != NULL)
        {
            m_higher->dot_edges(stream);
            stream << "\t" << m_value << " -> " << m_higher->m_value << "\n";
        }
    }
    void Node::dot_node_descrs(std::ostream & stream)
    {
        if(m_lower != NULL)
        {
            m_lower->dot_node_descrs(stream);
        }
        stream << "\t" << m_value << "[label=\"" << m_value << " (" << m_color <<")\" ]\n";
        if(m_higher != NULL)
        {
            m_higher->dot_node_descrs(stream);
        }
    }
inline bool Node::has_red_child(void)
{
    if(m_lower != NULL && m_lower->m_color == NODE_RED)
    {
        return true;
    }
    if(m_higher != NULL && m_higher->m_color == NODE_RED)
    {
        return true;
    }
    return false;
}

int Node::check_rbt_pathlength(void)
{
    int lleft = 1, lright = 1;
    if(m_lower != NULL)
    {
        lleft = m_lower->check_rbt_pathlength();
        if(lleft == -1)
        {
            return -1;
        }
    }
    if(m_higher != NULL)
    {
        lright = m_higher->check_rbt_pathlength();
        if(lright == -1)
        {
            return -1;
        }
    }
    if(lleft != lright)
    {
        return -1;
    }

    if(m_color == NODE_BLACK)
    {
        return lleft + 1;
    }
    return lleft;
}

int RBTree::rbt_pathlength(void)
{
    if(m_root == NULL)
    {
        return 1;
    }
    return m_root->check_rbt_pathlength();
}

    void RBTree::to_dot(std::ostream & stream)
    {
        if(m_root != NULL)
        {
            stream << "digraph g{\n";
            m_root->dot_node_descrs(stream);
            stream << "\n";
            m_root->dot_edges(stream);
            stream << "}\n";
        }
    }
    bool RBTree::has_value(int value)
    {
        Node * c_node = m_root;
        while(c_node != NULL)
        {
            if(c_node->m_value < value)
            {
                c_node = c_node->m_higher;
                continue;
            }
            if(c_node->m_value > value)
            {
                c_node = c_node->m_lower;
                continue;
            }
            return true;
        }
        return false;
    }
    inline void RBTree::left_rotate(Node * B)
    {
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
        Node * C = B->m_parent; // C
        Node * greatgrandparent = C->m_parent; // C's B
        Node * gamma = B->m_lower; // B's left child
        bool C_was_lower_child = C->is_lower_child();

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
            if(C_was_lower_child)
            {
                greatgrandparent->m_lower = B; // (c)
            }
            else
            {
                greatgrandparent->m_higher = B; // (c)
            }
        }
        B->m_parent = greatgrandparent; // (c)
    }
    inline void RBTree::right_rotate(Node * B)
    {
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
        Node * C = B->m_parent; // C
        Node * greatgrandparent = C->m_parent; // C's parent
        Node * gamma = B->m_higher; // B's right child
        bool C_was_lower_child = C->is_lower_child();

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
            if(C_was_lower_child)
            {
                greatgrandparent->m_lower = B; // (c)
            }
            else
            {
                greatgrandparent->m_higher = B; // (c)
            }
        }
        B->m_parent = greatgrandparent; //(c)
    }
    void RBTree::delete_value(int value)
    {
        Node * c_node = m_root;
        while(c_node != NULL)
        {
            if(c_node->m_value < value)
            {
                c_node = c_node->m_higher;
                continue;
            }
            if(c_node->m_value > value)
            {
                c_node = c_node->m_lower;
                continue;
            }

            delete_this_node(c_node);
            return;
        }
        throw std::runtime_error("value not in tree, cannot be deleted.");
    }
    inline void RBTree::delete_this_node(Node * c_node)
    {
        if(c_node->m_lower != NULL && c_node->m_higher != NULL)
        {
            Node * precessor = c_node->m_lower;

            // Find the inorder precessor.
            while(precessor->m_higher != NULL)
            {
                precessor = precessor->m_higher;
            }

            // Swap the values:
            c_node->m_value = precessor->m_value;
            // No need to update precessor, it will get deleted now.
            c_node = precessor;
        }

        // The only non-NULL child or NULL.
        Node * replace = NULL;
        if(c_node->m_lower != NULL)
        {
            replace = c_node->m_lower;
        }
        if(c_node->m_higher != NULL)
        {
            replace = c_node->m_higher;
        }

        // Replace c_node by its non-NULL child in the tree.
        Node * sibling = NULL;
        if(c_node->m_parent == NULL)
        {
            // Notify the tree.
            m_root = replace;
        }
        else
        {
            // Notify the parent.
            if(c_node->is_lower_child())
            {
                c_node->m_parent->m_lower = replace;
                sibling = c_node->m_parent->m_higher;
            }
            else
            {
                c_node->m_parent->m_higher = replace;
                sibling = c_node->m_parent->m_lower;
            }
        }
        if(replace != NULL)
        {
            // Notify the replacement.
            replace->m_parent = c_node->m_parent;
        }

        // Delete the node.
        char color = c_node->m_color;
        Node * parent = c_node->m_parent;
        delete c_node;

        // Check if we need to repair the RB property.
        if(color == NODE_RED)
        {
            return;
        }

        // c_node was black.
        if(replace != NULL && replace->m_color == NODE_RED)
        {
            replace->m_color = NODE_BLACK;
            return;
        }

        // We need to repair (replace is double-black).  Because replace might
        // be NULL we can't do that in an extra function.  Note that parent is
        // known since we stored it before deleting c_node.
        c_node = replace; // Use the name c_node again for better readability.
        while(1)
        {
            // No need to repair the root.
            if(parent == NULL)
            {
                //c_node->m_color = NODE_BLACK;
                return;
            }

            // We have a parent.
            Node * sibling = NULL;
            if(parent->m_lower == c_node)
            {
                sibling = parent->m_higher;
            }
            else
            {
                sibling = parent->m_lower;
            }

            // Cannot happen. If the c_node is double black
            // and its sibling is singly black and has no children we started
            // from an invalid RBT.
            //if(sibling == NULL)
            //{
            //    throw std::runtime_error("uncompensated double-black found");
            //}

            if(sibling->m_color == NODE_RED)
            {
                if(sibling->is_lower_child())
                {
                    right_rotate(sibling);
                }
                else
                {
                    left_rotate(sibling);
                }
                sibling->m_color = NODE_BLACK;
                parent->m_color = NODE_RED;

                continue;
            }

            if(sibling->has_red_child())
            {
                char parent_color = parent->m_color;
                parent->m_color = NODE_BLACK;
                if(sibling->is_lower_child())
                {
                    if(sibling->m_lower != NULL && sibling->m_lower->m_color == NODE_RED)
                    {
                        sibling->m_lower->m_color = NODE_BLACK;
                        right_rotate(sibling);
                        sibling->m_color = parent_color;
                        return;
                    }
                    else
                    {
                        sibling->m_higher->m_color = NODE_BLACK;
                        Node * new_sibling = sibling->m_higher;
                        left_rotate(sibling->m_higher);
                        right_rotate(new_sibling); // This is the new sibling.
                        new_sibling->m_color = parent_color;
                        return;
                    }
                }
                else
                {
                    if(sibling->m_higher != NULL && sibling->m_higher->m_color == NODE_RED)
                    {
                        sibling->m_higher->m_color = NODE_BLACK;
                        left_rotate(sibling);
                        sibling->m_color = parent_color;
                        return;
                    }
                    else
                    {
                        sibling->m_lower->m_color = NODE_BLACK;
                        Node * new_sibling = sibling->m_lower;
                        right_rotate(sibling->m_lower);
                        left_rotate(new_sibling); // This is the new sibling.
                        new_sibling->m_color = parent_color;
                        return;
                    }

                }

            }

            // Sibling is black and has only black children.
            sibling->m_color = NODE_RED;
            if(parent->m_color == NODE_RED)
            {
                // Change parent color from red to black. This
                // fixes the RBT properties.
                parent->m_color = NODE_BLACK;
                return;
            }
            // Parent is now double-black.
            c_node = parent;
            parent = c_node->m_parent;
            continue;
        }
    }

    void RBTree::export_inorder_iterative(std::vector<int> & vect)
    {
        vect.resize(0);
        if(m_root == NULL)
        {
            return;
        }

        repair_markers_if_needed();
        Node * c_node = m_root;
        m_marker_sanity = 0;
        while(c_node != NULL)
        {
            if((c_node->m_marker ^ m_marker_mask) == 0)
            {
                // 1st time visiting the node.
                c_node->m_marker ^= 0b01;
                if(c_node->m_lower != NULL)
                {
                    c_node = c_node->m_lower;
                }
                continue;
            }
            if((c_node->m_marker ^ m_marker_mask) == 0b01)
            {
                // 2nd time visiting the node.

                vect.push_back(c_node->m_value);
                c_node->m_marker ^= 0b10;
                if(c_node->m_higher != NULL)
                {
                    c_node = c_node->m_higher;
                }
                continue;
            }
            c_node = c_node->m_parent;
        }
        m_marker_sanity = 1;
        m_marker_mask ^= 0b11;
        
    }
}
