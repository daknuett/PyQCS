#pragma once
#include <iostream>
#include <vector>

namespace ll
{
    class Node
    {
        private:
        Node * m_next;
        int m_value;
        public:
        Node(int value);
        Node(int value, Node * next);
        void print(void);
        friend class SortedList;

    };
    class SortedList
    {
        private:
        Node * m_first;
        public:
        void insert(int value);
        bool has_value(int value);
        void print(void);
        void delete_value(int value);
        SortedList(void);
        ~SortedList(void);
        void export_vector(std::vector<int> & vect);

    };
}
