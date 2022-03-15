#ifndef SOFA_CSR_TO_ADJ_H
#define SOFA_CSR_TO_ADJ_H

#include <SofaSparseSolver/config.h>
#include <SofaBaseLinearSolver/MatrixLinearSolver.h>

extern "C" {
#include <metis.h>
}

namespace sofa::component::linearsolver
{
void CSR_to_adj(int n,int * M_colptr,int * M_rowind,type::vector<int> adj,type::vector<int> xadj); // compute the adjency matrix in CSR format from the matrix given in CSR format
}
#endif