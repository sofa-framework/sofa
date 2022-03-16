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
#ifndef SOFA_COMPONENT_LINEARSOLVER_SparseLUSolver_INL
#define SOFA_COMPONENT_LINEARSOLVER_SparseLUSolver_INL

#include <SofaSparseSolver/SparseLUSolver.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

using namespace sofa::defaulttype;
using namespace sofa::core::behavior;
using namespace sofa::simulation;
using namespace sofa::core::objectmodel;
using sofa::helper::system::thread::CTime;
using sofa::helper::system::thread::ctime_t;
using std::cerr;
using std::endl;

template<class TMatrix, class TVector,class TThreadManager>
SparseLUSolver<TMatrix,TVector,TThreadManager>::SparseLUSolver()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , f_tol( initData(&f_tol,0.001,"tolerance","tolerance of factorization") )
    , d_applyPermutation(initData(&d_applyPermutation, true ,"applyPermutation", "If true the solver will apply a fill-reducing permutation to the matrix of the system."))
{
}


template<class TMatrix, class TVector,class TThreadManager>
void SparseLUSolver<TMatrix,TVector,TThreadManager>::solve (Matrix& M, Vector& x, Vector& b)
{
    SparseLUInvertData<Real> * invertData = (SparseLUInvertData<Real>*) this->getMatrixInvertData(&M);
    int n = invertData->A.n;

    sofa::helper::ScopedAdvancedTimer solveTimer("solve");

    cs_pvec (n, invertData->perm.data() , b.ptr(), invertData->tmp) ; // x = P*b
    cs_lsolve (invertData->N->L, invertData->tmp) ;		// x = L\x 
    cs_usolve (invertData->N->U, invertData->tmp) ;		// x = U\x 
    cs_pvec (n, invertData->iperm.data() , invertData->tmp, x.ptr()) ;	// b = Q*x 
    
}

template<class TMatrix, class TVector,class TThreadManager>
void SparseLUSolver<TMatrix,TVector,TThreadManager>::invert(Matrix& M)
{
    SparseLUInvertData<Real> * invertData = (SparseLUInvertData<Real>*) this->getMatrixInvertData(&M);

    if (invertData->S) cs_sfree(invertData->S);
    if (invertData->N) cs_nfree(invertData->N);
    if (invertData->tmp) cs_free(invertData->tmp);
    M.compress();
    //remplir A avec M
    invertData->A.nzmax = M.getColsValue().size();	// maximum number of entries
    invertData->A.m = M.rowBSize();					// number of rows
    invertData->A.n = M.colBSize();					// number of columns
    invertData->A_p = M.getRowBegin();
    invertData->A.p = (int *) &(invertData->A_p[0]);							// column pointers (size n+1) or col indices (size nzmax)
    invertData->A_i = M.getColsIndex();
    invertData->A.i = (int *) &(invertData->A_i[0]);							// row indices, size nzmax
    invertData->A_x = M.getColsValue();
    invertData->A.x = (Real *) &(invertData->A_x[0]);				// numerical values, size nzmax
    invertData->A.nz = -1;							// # of entries in triplet matrix, -1 for compressed-col
    cs_dropzeros( &invertData->A );

    invertData->perm.resize(invertData->A.n);
    invertData->iperm.resize(invertData->A.n);

    invertData->tmp = (Real *) cs_malloc (invertData->A.n, sizeof (Real)) ;

    fill_reducing_perm(invertData->A, invertData->perm.data(), invertData->iperm.data() ); // compute the fill reducing permutation

    invertData->permuted_A = cs_permute(&(invertData->A), invertData->iperm.data(), invertData->perm.data(), 1); 
    invertData->S = symbolic_LU( invertData->permuted_A );

    sofa::helper::ScopedAdvancedTimer factorizationTimer("factorization");
    invertData->N = cs_lu ( invertData->permuted_A, invertData->S, f_tol.getValue()) ;		/* numeric LU factorization */
}


template<class TMatrix, class TVector,class TThreadManager>
void SparseLUSolver<TMatrix,TVector,TThreadManager>::fill_reducing_perm(cs A,int * perm,int * invperm)
{
    sofa::helper::ScopedAdvancedTimer permTimer("permutation");
    
    int n = A.n;
    if(d_applyPermutation.getValue() )
    {
        int *M_colptr=A.p, *M_rowind=A.i ;
        type::vector<int> xadj,adj,tran_countvec,t_xadj,t_adj;

/*  
    //old version
        xadj.resize(n+1);
        adj.resize(M_colptr[n]-n);

        int it=0;
        for(int j=0; j<n; j++)
        {
            xadj[j] = M_colptr[j] - j ;

            for(int ip = M_colptr[j]; ip<M_colptr[j+1];ip++)
            {
               int i = M_rowind[ip];
               if(i !=j ) adj[it++] = i;
            }
        }
        xadj[n] = M_colptr[n] - n;
*/

 //Compute transpose in tran_colptr, tran_rowind, tran_values, tran_D
            tran_countvec.clear();
            tran_countvec.resize(n);

            //First we count the number of value on each row.
            for (int j=0;j<n;j++) {
                for (int i=M_colptr[j];i<M_colptr[j+1];i++) {
                    int col = M_rowind[i];
                    if (col>j) tran_countvec[col]++;
                }
            }

            //Now we make a scan to build tran_colptr
            t_xadj.resize(n+1);
            t_xadj[0] = 0;
            for (int j=0;j<n;j++) t_xadj[j+1] = t_xadj[j] + tran_countvec[j];

            //we clear tran_countvec because we use it now to store hown many values are written on each line
            tran_countvec.clear();
            tran_countvec.resize(n);

            t_adj.resize(t_xadj[n]);
            for (int j=0;j<n;j++) {
            for (int i=M_colptr[j];i<M_colptr[j+1];i++) {
                int line = M_rowind[i];
                if (line>j) {
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
                //copy the lower part
                for (int ip = t_xadj[j]; ip < t_xadj[j+1]; ip++) {
                    adj.push_back(t_adj[ip]);
                }

                //copy only the upper part
                for (int ip = M_colptr[j]; ip < M_colptr[j+1]; ip++) {
                    int col = M_rowind[ip];
                    if (col > j) adj.push_back(col);
                }

                xadj[j+1] = adj.size();
            }

        METIS_NodeND(&n, xadj.data(), adj.data(), NULL, NULL, perm,invperm);

    }
    else
    {
        for(int j=0;j<n;j++)
        {
            perm[j] = j;
            invperm[j] = j;
        }
    }

}

template<class TMatrix, class TVector,class TThreadManager>
css* SparseLUSolver<TMatrix,TVector,TThreadManager>::symbolic_LU(cs *A)
{// based on cs_sqr
    
    int n;
    css *S ;
    if (!A) return (NULL) ;		    /* check inputs */
    n = A->n ;
    S = (css*)cs_calloc (1, sizeof (css)) ;	    /* allocate symbolic analysis */
    if (!S) return (NULL) ;		    /* out of memory */  
	S->unz = 4*(A->p [n]) + n ;	    /* for LU factorization only, */
	S->lnz = S->unz ;		    /* guess nnz(L) and nnz(U) */

    return S ;
}

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
