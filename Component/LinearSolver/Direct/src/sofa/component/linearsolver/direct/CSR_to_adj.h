#pragma once

#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>

namespace sofa::component::linearsolver
{
void CSR_to_adj(int n,int * M_colptr,int * M_rowind,type::vector<int>& adj,type::vector<int>& xadj,type::vector<int>& t_adj, type::vector<int>& t_xadj, type::vector<int>& tran_countvec ); 
/* compute the adjency matrix in CSR format from the matrix given in CSR format

M_colptr[i+1]-M_colptr[i] is the number of non null values on the i-th line of the matrix
M_rowind[M_colptr[i]] to M_rowind[M_colptr[i+1]] is the list of the indices of the columns containing a non null value on the i-th line

xadj[i+1]-xadj[i] is the number of neighbors of the i-th node
adj[xadj[i]] is the first neighbor of the i-th node

*/
}

