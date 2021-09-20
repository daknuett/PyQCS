#pragma once

#include <rbt/rbt.hpp>
#include <vector>
#include <map>

namespace graphical
{
    /**
    * The pauli operators as used by the GraphState::project_to and
    * GraphState::measurement_probability methods. This map is purely for exporting
    * the symbols.
    * */
    static const std::map<const std::string, const uint8_t> pauli_operators = 
    {
        {"Z", 0 }, {"Y", 1}, {"X", 2}, {"-Z", 3}, {"-Y", 4}, {"-X", 5}
    };
    static uint8_t const pauli_Z = 0;
    static uint8_t const pauli_Y = 1;
    static uint8_t const pauli_X = 2;
    static uint8_t const pauli_mZ = 3;
    static uint8_t const pauli_mY = 4;
    static uint8_t const pauli_mX = 5;
    class GraphState
    {
        private:
        size_t m_nqbits;
        std::vector<int> m_vops;
        std::vector<rbt::RBTree> m_ngbhds;

        inline void do_project_to(int i, int pauli);
        inline void toggle_edge(int i, int j);
        inline void isolated_two_qbit_CZ(int i, int j);
        inline bool can_clear_vop(int i, int j);
        inline void clear_vop(int i, int j);
        inline void La_transformation(int i, int repeat);
        inline void toggle_neighbourhood(int i);
        inline void isolate_qbit(int i);

        /*
        * Qbits i, j are at most connected to each other.
        * */
        bool qbits_are_isolated(int i, int j);

        public:
        GraphState(size_t nqbits);
        ~GraphState();
        GraphState(GraphState & orig);
        GraphState(GraphState const & orig);
        void apply_CL(int i, int vop);
        void apply_CZ(int i, int j);
        /**
        * Returns the probability for the measurement of Pauli observable ``pauli''.
        * Results should be interpreted as such:
        * 
        * Result -1: Probablility is zero.
        * Result 0: Probability is (1/2)^0 = 1.
        * Result 1: Probablility is (1/2)^1 = 1/2.
        * */
        int measurement_probability(int i, int pauli);
        void project_to(int i, int pauli);
        void export_to_vectors(std::vector<int> & vops, std::vector<std::vector<int>> & ngbs);
        int nqbits(void);

        /*
        * Returns the overlap of the graphical states ``this'' and ``other''.
        * The return format is similar to ``measurement_probability'':
        *
        * Result k >= 0: Overlap is (1/srqrt(2)) ^ k
        * Result k = -1: Overlap is 0.
        * */
        int operator*(GraphState & other);
    };
}
