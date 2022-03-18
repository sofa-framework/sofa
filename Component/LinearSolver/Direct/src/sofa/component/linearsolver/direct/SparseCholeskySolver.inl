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
    , type_perm(initData(&type_perm, "permutation", "Type of fill reducing permutation"))
{
    computePermutation = true;
    sofa::helper::OptionsGroup type_permOptions(3,"None", "SuiteSparse", "METIS");
    type_permOptions.setSelectedItem(1); // default SuiteSparse
    type_perm.setValue(type_permOptions);
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
    
    switch( type_perm.getValue().getSelectedId() )
    {
        case 0://None->identity
        default:
            cs_ipvec (n, S->Pinv, r , (double*) tmp.data() );	//used here to copy, Pinv = Id
            cs_lsolve (N->L, (double*) tmp.data() );			//x = L\x
            cs_ltsolve (N->L, (double*) tmp.data() );			//x = L'\x/
            cs_pvec (n, S->Pinv, (double*) tmp.data() , z );	 //used here to copy, transopse(Pinv) = Id
            break;
    
        case 1://SuiteSparse
            
            cs_ipvec (n, S->Pinv,  r , (double*) tmp.data() );	//x = P*b , permutation on rows
            cs_lsolve (N->L, (double*) tmp.data() );			//x = L\x
            cs_ltsolve (N->L, (double*) tmp.data() );			//x = L'\x/
            cs_pvec (n, S->Pinv, (double*) tmp.data() , z );	 //b = P'*x , permutation on columns
            break;

        case 2://METIS

            cs_ipvec (n, perm.data(),  r , tmp.data() );	//x = P*b , permutation on rowsl;
            cs_lsolve (N->L, tmp.data() );			//x = L\x
            cs_ltsolve (N->L, tmp.data() );			//x = L'\x/
            cs_pvec (n, perm.data() , tmp.data() , z );	 //b = P'*x , permutation on columns
            break;

    }
}


template<class TMatrix, class TVector>
void SparseCholeskySolver<TMatrix,TVector>::solveT(float * z, float * r)
{
    int n = A.n;

    switch( type_perm.getValue().getSelectedId() )
    {
        case 0://None->identity
        default:
            cs_ipvec (n, S->Pinv, (double*) r , (double*) tmp.data() );	//used here to copy, Pinv = Id
            cs_lsolve (N->L, (double*) tmp.data() );			//x = L\x
            cs_ltsolve (N->L, (double*) tmp.data() );			//x = L'\x/
            cs_pvec (n, S->Pinv, (double*) tmp.data() , (double*) z );	 //used here to copy, transopse(Pinv) = Id
            break;
    
        case 1://SuiteSparse
            
            cs_ipvec (n, S->Pinv, (double*) r , (double*) tmp.data() );	//x = P*b , permutation on rows
            cs_lsolve (N->L, (double*) tmp.data() );			//x = L\x
            cs_ltsolve (N->L, (double*) tmp.data() );			//x = L'\x/
            cs_pvec (n, S->Pinv, (double*) tmp.data() , (double*) z );	 //b = P'*x , permutation on columns
            break;

        case 2://METIS
            cs_pvec (n, perm.data(),  (double*) r , (double*) tmp.data() );	//x = P*b , permutation on rows
            cs_lsolve (N->L, (double*) tmp.data() );			//x = L\x
            cs_ltsolve (N->L, (double*) tmp.data() );			//x = L'\x/
            cs_ipvec (n, perm.data() , (double*) tmp.data() , (double*) z );	 //b = P'*x , permutation on columns
            break;

    }
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

    switch (type_perm.getValue().getSelectedId() )
    {
        case 0:
        default://None->identity
            {
                sofa::helper::ScopedAdvancedTimer factorization_permTimer("factorization_perm");
                S = symbolic_Chol (&A) ;		/* ordering and symbolic analysis */
                N = cs_chol (&A, S) ;		/* numeric Cholesky factorization */
                break;
            }
        case 1://SuiteSparse
            {
                sofa::helper::ScopedAdvancedTimer factorization_permTimer("factorization_perm");
                int order = -1;
                S = cs_schol (&A, order) ;		/* ordering and symbolic analysis */
                N = cs_chol (&A, S) ;		/* numeric Cholesky factorization */
            }
            break;
        case 2://METIS
            perm.resize(A.n);
            iperm.resize(A.n);
            { 
                sofa::helper::ScopedAdvancedTimer factorization_permTimer("factorization_perm");
                if(computePermutation)
                {
                    fill_reducing_perm(A , perm.data(), iperm.data() ); // compute the fill reducing permutation
                    computePermutation = false;
                }
                permuted_A = cs_permute( &A , perm.data() , iperm.data() , 1);
                S = symbolic_Chol( permuted_A ); // symbolic analysis  
                N = cs_chol (permuted_A, S) ;		/* numeric Cholesky factorization */
            }

            break;
    }

}

template<class TMatrix, class TVector>
void SparseCholeskySolver<TMatrix,TVector>::fill_reducing_perm(cs A,int * perm,int * invperm)
{
    int n = A.n;
    sofa::type::vector<int> adj, xadj, t_adj, t_xadj, tran_countvec;
    CSR_to_adj( A.n, A.p , A.i , adj, xadj, t_adj, t_xadj, tran_countvec );
    METIS_NodeND(&n, xadj.data(), adj.data(), nullptr, nullptr, perm,invperm);

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
    // we do not use the permutation of SuiteSparse
    S->Q = nullptr ; // permutation on columns set to identity
    S->Pinv = nullptr; // permutation on rows set to identity
    cs_free (c) ;
    return ((S->lnz >= 0) ? S : cs_sfree (S)) ;
}


} // namespace sofa::component::linearsolver::direct
