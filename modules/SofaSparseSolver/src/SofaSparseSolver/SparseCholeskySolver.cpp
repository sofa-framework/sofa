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
#define SOFA_COMPONENT_LINEARSOLVER_SPARSECHOLESKYSOLVER_CPP
#include <SofaSparseSolver/SparseCholeskySolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/ScopedAdvancedTimer.h>


namespace sofa::component::linearsolver
{

using namespace sofa::defaulttype;
using namespace sofa::core::behavior;
using namespace sofa::simulation;
using namespace sofa::core::objectmodel;
using sofa::helper::system::thread::CTime;
using sofa::helper::system::thread::ctime_t;

template<class TMatrix, class TVector>
SparseCholeskySolver<TMatrix,TVector>::SparseCholeskySolver()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , S(nullptr), N(nullptr)
    , d_applyPermutation(initData(&d_applyPermutation, true ,"applyPermutation", "If true the solver will apply a fill-reducing permutation to the matrix of the system."))
{
}

template<class TMatrix, class TVector>
SparseCholeskySolver<TMatrix,TVector>::~SparseCholeskySolver()
{
    if (S) cs_sfree (S);
    if (N) cs_nfree (N);
}

template<class TMatrix, class TVector>
void SparseCholeskySolver<TMatrix,TVector>::solveT(double * z, double * r)
{
    int n = A.n;

    cs_ipvec (n, S->Pinv, r, (double*) &(tmp[0]));	//x = P*b

    cs_lsolve (N->L, (double*) &(tmp[0]));			//x = L\x
    cs_ltsolve (N->L, (double*) &(tmp[0]));			//x = L'\x/

    cs_pvec (n, S->Pinv, (double*) &(tmp[0]), z);	 //b = P'*x
}

template<class TMatrix, class TVector>
void SparseCholeskySolver<TMatrix,TVector>::solveT(float * z, float * r)
{
    int n = A.n;
    z_tmp.resize(n);
    r_tmp.resize(n);
    for (int i=0; i<n; i++) r_tmp[i] = (double) r[i];

    sofa::helper::AdvancedTimer::stepBegin("solve");
    cs_pvec(n, perm.data(), r_tmp.data() ,tmp.data() );
    cs_lsolve (N->L, tmp.data() );			//x = L\x
    cs_ltsolve (N->L, tmp.data() );			//x = L'\x/
    cs_pvec( n, iperm.data() , tmp.data() , z_tmp.data() );
    sofa::helper::AdvancedTimer::stepEnd("solve");
    
    for (int i=0; i<n; i++) z[i] = (float) z_tmp[i];
}


template<class TMatrix, class TVector>
void SparseCholeskySolver<TMatrix,TVector>::solve (Matrix& /*M*/, Vector& z, Vector& r)
{
    solveT(z.ptr(),r.ptr());
}

template<class TMatrix, class TVector>
void SparseCholeskySolver<TMatrix,TVector>::invert(Matrix& M)
{
    if (S) cs_sfree(S);
    if (N) cs_nfree(N);
    M.compress();

    A.nzmax = M.getColsValue().size();	// maximum number of entries
    A_p = (int *) &(M.getRowBegin()[0]);
    A_i = (int *) &(M.getColsIndex()[0]);
    A_x.resize(A.nzmax);
    for (int i=0; i<A.nzmax; i++) A_x[i] = (double) M.getColsValue()[i];
    //remplir A avec M
    A.m = M.rowBSize();					// number of rows
    A.n = M.colBSize();					// number of columns
    A.p = A_p;							// column pointers (size n+1) or col indices (size nzmax)
    A.i = A_i;							// row indices, size nzmax
    A.x = (double*) &(A_x[0]);				// numerical values, size nzmax
    A.nz = -1;							// # of entries in triplet matrix, -1 for compressed-col
    cs_dropzeros( &A );
    tmp.resize(A.n);
    
    perm.resize(A.n);
    iperm.resize(A.n);

    fill_reducing_perm(A , perm.data(), iperm.data() ); // compute the fill reducing permutation
    S = symbolic_Chol( &A ); // symbolic analysis
    permuted_A = cs_permute( &(A), iperm.data() , perm.data() , 1);

    sofa::helper::AdvancedTimer::stepBegin("factorization");
    N = cs_chol (&A, S) ;		/* numeric Cholesky factorization */
    sofa::helper::AdvancedTimer::stepEnd("factorization");
}

template<class TMatrix, class TVector>
void SparseCholeskySolver<TMatrix,TVector>::fill_reducing_perm(cs A,int * perm,int * invperm)
{
    int n = A.n;
    if(d_applyPermutation.getValue() )
    {
        int *M_colptr=A.p, *M_rowind=A.i ;
        type::vector<int> xadj,adj,tran_countvec,t_xadj,t_adj;
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

template<class TMatrix, class TVector>
css* SparseCholeskySolver<TMatrix,TVector>::symbolic_Chol(cs *A)
{ //based on cs_chol
    int n, *c, *post;
    cs *C ;
    css *S ;
    if (!A) return (NULL) ;		    /* check inputs */
    n = A->n ;
    S = (css*)cs_calloc (1, sizeof (css)) ;	    /* allocate symbolic analysis */
    if (!S) return (NULL) ;		    /* out of memory */
    C = cs_symperm (A, S->Pinv, 0) ;	    /* C = spones(triu(A(P,P))) */
    S->parent = cs_etree (C, 0) ;	    /* find etree of C */
    post = cs_post (n, S->parent) ;	    /* postorder the etree */
    c = cs_counts (C, S->parent, post, 0) ; /* find column counts of chol(C) */
    cs_free (post) ;
    cs_spfree (C) ;
    S->cp = (int*)cs_malloc (n+1, sizeof (int)) ; /* find column pointers for L */
    S->unz = S->lnz = cs_cumsum (S->cp, c, n) ;
    cs_free (c) ;
    return ((S->lnz >= 0) ? S : cs_sfree (S)) ;
}




using namespace sofa::linearalgebra;

int SparseCholeskySolverClass = core::RegisterObject("Direct linear solver based on Sparse Cholesky factorization, implemented with the CSPARSE library")
        .add< SparseCholeskySolver< CompressedRowSparseMatrix<double>,FullVector<double> > >(true)
        .add< SparseCholeskySolver< CompressedRowSparseMatrix<float>,FullVector<float> > >()
        ;

template class SOFA_SOFASPARSESOLVER_API SparseCholeskySolver< CompressedRowSparseMatrix<double>,FullVector<double> >;
template class SOFA_SOFASPARSESOLVER_API SparseCholeskySolver< CompressedRowSparseMatrix<float>,FullVector<float> >;

} // namespace sofa::component::linearsolver
