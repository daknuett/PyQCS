#pragma once

#include <rbt/rbt.hpp>
#include <vector>

namespace graphical
{
    class GraphState
    {
        private:
        size_t m_nqbits;
        int * m_vops;
        rbt::RBTree * m_ngbhds;

        void do_project_to(int i, int pauli);
        void after_vop_project_to(int i, int pauli);
        inline void toggle_edge(int i, int j);
        inline void isolated_two_qbit_CZ(int i, int j);

        /*
        * Qbits i, j are at most connected to each other.
        * */
        bool qbits_are_isolated(int i, int j);

        public:
        GraphState(size_t nqbits);
        ~GraphState();
        GraphState(GraphState & orig);
        void apply_CL(int i, int vop);
        void apply_CZ(int i, int j);
        int measurement_probability(int i, int pauli);
        void project_to(int i, int pauli);
        void export_to_vectors(std::vector<int> & vops, std::vector<std::vector<int>> & ngbs);

    };
}
