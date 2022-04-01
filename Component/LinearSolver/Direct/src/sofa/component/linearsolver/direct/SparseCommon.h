#pragma once

#include <sofa/component/linearsolver/direct/config.h>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>
#include <csparse.h>

extern "C" {
#include <metis.h>
}

namespace sofa::component::linearsolver
{
void CSR_to_adj(int n,int * M_colptr,int * M_rowind,type::vector<int>& adj,type::vector<int>& xadj,type::vector<int>& t_adj, type::vector<int>& t_xadj, type::vector<int>& tran_countvec ); 
/** 
compute the adjency matrix in CSR format from the matrix given in CSR format

M_colptr[i+1]-M_colptr[i] is the number of non null values on the i-th line of the matrix
M_rowind[M_colptr[i]] to M_rowind[M_colptr[i+1]] is the list of the indices of the columns containing a non null value on the i-th line

xadj[i+1]-xadj[i] is the number of neighbors of the i-th node
adj[xadj[i]] is the first neighbor of the i-th node

**/

void fill_reducing_perm(const cs &A,int * perm,int * invperm); /// compute the fill reducing permutation via METIS

inline bool compareMatrixShape(int s_M, int * M_colptr,int * M_rowind, int s_P, int * P_colptr,int * P_rowind) {
    if (s_M != s_P) return true;
    if (M_colptr[s_M] != P_colptr[s_M] ) return true;

    for (int i=0;i<s_P;i++) {
        if (M_colptr[i]!=P_colptr[i]) return true;
    }

    for (int i=0;i<M_colptr[s_M];i++) {
        if (M_rowind[i]!=P_rowind[i]) return true;
    }

    return false;
}



}

