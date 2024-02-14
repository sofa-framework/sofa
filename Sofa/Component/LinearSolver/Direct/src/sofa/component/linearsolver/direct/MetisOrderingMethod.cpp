/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/linearsolver/direct/MetisOrderingMethod.h>
#include <sofa/core/ObjectFactory.h>

extern "C" {
    #include <metis.h>
}

namespace sofa::component::linearsolver::direct
{

int MetisOrderingMethodClass = sofa::core::RegisterObject("Nested dissection algorithm implemented in the METIS library.")
    .add<MetisOrderingMethod>();

std::string MetisOrderingMethod::methodName() const
{
    return "Metis";
}

void csrToAdjMETIS(int n, int * M_colptr, int * M_rowind, type::vector<int>& adj, type::vector<int>& xadj , type::vector<int>& t_adj , type::vector<int>& t_xadj, type::vector<int>& tran_countvec)
{
    //Compute transpose in tran_colptr, tran_rowind, tran_values, tran_D
    tran_countvec.clear();
    tran_countvec.resize(n);

    // First we count the number of value on each row.
    for (int j=0;j<n;j++)
    {
        for (int i=M_colptr[j];i<M_colptr[j+1];i++)
        {
            const int col = M_rowind[i];
            if (col>j) tran_countvec[col]++;
        }
    }

    // Now we make a scan to build tran_colptr
    t_xadj.resize(n+1);
    t_xadj[0] = 0;
    for (int j=0;j<n;j++) t_xadj[j+1] = t_xadj[j] + tran_countvec[j];

    // we clear tran_countvec because we use it now to store hown many values are written on each line
    tran_countvec.clear();
    tran_countvec.resize(n);

    t_adj.resize(t_xadj[n]);
    for (int j=0;j<n;j++)
    {
        for (int i=M_colptr[j];i<M_colptr[j+1];i++)
        {
            const int line = M_rowind[i];
            if (line>j)
            {
                t_adj[t_xadj[line] + tran_countvec[line]] = j;
                tran_countvec[line]++;
            }
        }
    }

    adj.clear();
    xadj.resize(n+1);
    xadj[0] = 0;
    for (int j=0; j<n; j++)
    {
        // copy the lower part
        for (int ip = t_xadj[j]; ip < t_xadj[j+1]; ip++)
        {
            adj.push_back(t_adj[ip]);
        }

        // copy only the upper part
        for (int ip = M_colptr[j]; ip < M_colptr[j+1]; ip++)
        {
            int col = M_rowind[ip];
            if (col > j) adj.push_back(col);
        }

        xadj[j+1] = adj.size();
    }
}

void MetisOrderingMethod::computePermutation(
    const SparseMatrixPattern& inPattern, int* outPermutation,
    int* outInversePermutation)
{
    type::vector<int> xadj,adj,t_xadj,t_adj;
    type::vector<int> tran_countvec;
    csrToAdjMETIS(inPattern.matrixSize, inPattern.rowBegin, inPattern.colsIndex, adj, xadj, t_adj, t_xadj, tran_countvec );

    /*
    int numflag = 0, options = 0;
    The new API of metis requires pointers on numflag and "options" which are "structure" to parametrize the factorization
     We give NULL and NULL to use the default option (see doc of metis for details) !
     If you have the error "SparseLDLSolver failure to factorize, D(k,k) is zero" that probably means that you use the previsou version of metis.
     In this case you have to download and install the last version from : www.cs.umn.edu/~metis‎
    */
    idx_t n = inPattern.matrixSize;
    METIS_NodeND(&n, xadj.data(), adj.data(), nullptr, nullptr, outPermutation, outInversePermutation);
}

}
