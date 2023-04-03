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
#pragma once

#include <sofa/component/linearsolver/direct/SparseCholeskySolver.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

namespace sofa::component::linearsolver::direct
{

template<class TMatrix, class TVector>
SparseCholeskySolver<TMatrix,TVector>::SparseCholeskySolver()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , S(nullptr), N(nullptr)
    , d_typePermutation(initData(&d_typePermutation, "permutation", "Type of fill reducing permutation"))
{   
    sofa::helper::OptionsGroup d_typePermutationOptions{"None", "SuiteSparse", "METIS"};
    d_typePermutationOptions.setSelectedItem(0); // default None
    d_typePermutation.setValue(d_typePermutationOptions);
}

template<class TMatrix, class TVector>
SparseCholeskySolver<TMatrix,TVector>::~SparseCholeskySolver()
{
    if (S) cs_sfree (S);
    if (N) cs_nfree (N);
}

template<class TMatrix, class TVector>
void SparseCholeskySolver<TMatrix,TVector>::solve (Matrix& /*M*/, Vector& x, Vector& b)
{
    const int n = A.n;

    sofa::helper::ScopedAdvancedTimer solveTimer("solve");

    switch( d_typePermutation.getValue().getSelectedId() )
    {
    case 0://None->identity
    case 1://SuiteSparse
        if(N)
        {
            cs_ipvec (n, S->Pinv,  (double*)b.ptr() , tmp.data() );	//x = P*b , permutation on rows
            cs_lsolve (N->L, tmp.data() );			//x = L\x
            cs_ltsolve (N->L, tmp.data() );			//x = L'\x/
            cs_pvec (n, S->Pinv, tmp.data() , (double*)x.ptr() );	 //x = P'*x , permutation on columns
        }
        else
        {
            msg_error() << "Cannot solve system due to invalid factorization";
        }
        break;

    case 2://METIS
        if(N)
        {
            cs_ipvec (n, perm.data(),  (double*)b.ptr() , tmp.data() );	//x = P*b , permutation on rows
            cs_lsolve (N->L, tmp.data() );			//x = L\x
            cs_ltsolve (N->L, tmp.data() );			//x = L'\x/
            cs_pvec (n, perm.data() , tmp.data() , (double*)x.ptr() );	 //x = P'*x , permutation on columns
        }
        else
        {
            msg_error() << "Cannot solve system due to invalid factorization";
        }
        break;

    default:
        break;

    }

}

template<class TMatrix, class TVector>
void SparseCholeskySolver<TMatrix,TVector>::invert(Matrix& M)
{
    if (N) cs_nfree(N);
    M.compress();

    A.nzmax = M.getColsValue().size();	// maximum number of entries
    A_p = (int *) &(M.getRowBegin()[0]);
    A_i = (int *) &(M.getColsIndex()[0]);
    A_x.resize(A.nzmax);
    for (int i=0; i<A.nzmax; i++) A_x[i] = (double) M.getColsValue()[i];
    // build A with M
    A.m = M.rowBSize();					// number of rows
    A.n = M.colBSize();					// number of columns
    A.p = A_p;							// column pointers (size n+1) or col indices (size nzmax)
    A.i = A_i;							// row indices, size nzmax
    A.x = &(A_x[0]);				// numerical values, size nzmax
    A.nz = -1;							// # of entries in triplet matrix, -1 for compressed-col
    cs_dropzeros( &A );
    tmp.resize(A.n);

    {
        sofa::helper::ScopedAdvancedTimer factorization_permTimer("factorization_perm");

        notSameShape = compareMatrixShape( A.n , A.p , A.i, Previous_colptr.size()-1 , Previous_colptr.data() , Previous_rowind.data() );

        switch (d_typePermutation.getValue().getSelectedId() )
        {
            case 0:
            default:// None->identity
                suiteSparseFactorization(false);
                break;

            case 1:// SuiteSparse
                suiteSparseFactorization(true);
                break;

            case 2:// METIS
                if( notSameShape )
                {
                    perm.resize(A.n);
                    iperm.resize(A.n);

                    fillReducingPermutation( A , iperm.data(), perm.data() ); // compute the fill reducing permutation
                }

                permuted_A = cs_permute( &A , perm.data() , iperm.data() , 1);

                if ( notSameShape )
                {
                    if (S) cs_sfree(S);
                    S = symbolic_Chol( permuted_A );
                } // symbolic analysis

                N = cs_chol (permuted_A, S) ;		// numeric Cholesky factorization
                assert(N);

                cs_free(permuted_A);
                break;
        }
    }

    // store the shape of the matrix
    if ( notSameShape )
    {
        Previous_rowind.clear();
        Previous_colptr.resize(A.n +1);
        for(int i=0 ; i<A.n ; i++)
        {
            Previous_colptr[i+1] = A.p[i+1];

            for( int j=A.p[i] ; j < A.p[i+1] ; j++)
            {
                Previous_rowind.push_back(A.i[j]);
            }
        }
    }

}

template <class TMatrix, class TVector>
void SparseCholeskySolver<TMatrix, TVector>::suiteSparseFactorization(bool applyPermutation)
{
    if( notSameShape )
    {
        if (S)
        {
            cs_sfree(S);
        }
        const auto order = applyPermutation ? 0 : -1;
        S = cs_schol (&A, order);
    }
    assert(S);
    assert(S->cp);
    assert(S->parent);
    N = cs_chol (&A, S) ;		// numeric Cholesky factorization
    msg_error_when(!N) << "Matrix could not be factorized: possibly not positive-definite";
}

template<class TMatrix, class TVector>
css* SparseCholeskySolver<TMatrix,TVector>::symbolic_Chol(cs *A)
{ //based on cs_schol
    int n, *c, *post;
    cs *C ;
    css *S ;
    if (!A) return (NULL) ;		    // check inputs 
    n = A->n ;
    S = (css*)cs_calloc (1, sizeof (css)) ;	    // allocate symbolic analysis 
    if (!S) return (NULL) ;		    // out of memory 
    C = cs_symperm (A, S->Pinv, 0) ;	    // C = spones(triu(A(P,P))) 
    S->parent = cs_etree (C, 0) ;	    // find etree of C 
    post = cs_post (n, S->parent) ;	    // postorder the etree 
    c = cs_counts (C, S->parent, post, 0) ; // find column counts of chol(C) 
    cs_free (post) ;
    cs_spfree (C) ;
    S->cp = (int*)cs_malloc (n+1, sizeof (int)) ; // find column pointers for L 
    S->unz = S->lnz = cs_cumsum (S->cp, c, n) ;
    // we do not use the permutation of SuiteSparse
    S->Q = nullptr ; // permutation on columns set to identity
    S->Pinv = nullptr; // permutation on rows set to identity
    cs_free (c) ;
    return ((S->lnz >= 0) ? S : cs_sfree (S)) ;
}

} // namespace sofa::component::linearsolver::direct
