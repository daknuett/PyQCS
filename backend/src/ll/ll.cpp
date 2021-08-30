#include <stdexcept>
#include <ll/ll.hpp>
namespace ll
{
    Node::Node(int value): m_value(value)
    {
        m_next = NULL;
    }
    Node::Node(int value, Node * next): m_next(next), m_value(value){}
    void Node::print(void)
    {
        if(m_next != NULL)
        {
            std::cout << m_value << " -> ";
        }
        else
        {
            std::cout << m_value << " ";
        }
    }

    void SortedList::insert(int value)
    {
        Node * c_node = m_first;
        Node * p_node = NULL;
        while(c_node != NULL)
        {
            if(c_node->m_value < value)
            {
                p_node = c_node;
                c_node = c_node->m_next;
            }
            else
            {
                break;
            }
        }

        if(p_node == NULL)
        {
            m_first = new Node(value, c_node);
        }
        else
        {
            p_node->m_next = new Node(value, c_node);
        }
    }

    bool SortedList::has_value(int value)
    {
        Node * c_node = m_first;
        while(c_node != NULL)
        {
            if(c_node->m_value < value)
            {
                c_node = c_node->m_next;
                continue;
            }
            if(c_node->m_value > value)
            {
                break;
            }
            if(c_node->m_value == value)
            {
                return true;
            }
        }
        return false;
    }

    void SortedList::print(void)
    {
        Node * c_node = m_first;
        std::cout << "[- ";
        while(c_node != NULL)
        {
            c_node->print();
            c_node = c_node->m_next;
        }
        std::cout << "@\n";
    }
    void SortedList::delete_value(int value)
    {
        Node * c_node = m_first;
        Node * p_node = NULL;
        while(c_node != NULL)
        {
            if(c_node->m_value < value)
            {
                p_node = c_node;
                c_node = c_node->m_next;
                continue;
            }
            if(c_node->m_value == value)
            {
                if(p_node == NULL)
                {
                    m_first = c_node->m_next;
                }
                else
                {
                    p_node->m_next = c_node->m_next;
                }
                delete c_node;
                return;
            }
            break;
        }
        throw std::invalid_argument("value not in list");
    }
    SortedList::SortedList(void)
    {
        m_first = NULL;
    }
    SortedList::~SortedList(void)
    {
        Node * c_node = m_first;
        while(c_node != NULL)
        {
            Node * t_node = c_node->m_next;
            delete c_node;
            c_node = t_node;
        }
    }
    void SortedList::export_vector(std::vector<int> & vect)
    {
        vect.resize(0);
        Node * c_node = m_first;
        while(c_node != NULL)
        {
            vect.push_back(c_node->m_value);
            c_node = c_node->m_next;
        }
    }
}
