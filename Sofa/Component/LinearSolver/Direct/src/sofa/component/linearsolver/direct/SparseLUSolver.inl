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

#include <sofa/component/linearsolver/direct/SparseLUSolver.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

namespace sofa::component::linearsolver::direct
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
    , d_typePermutation(initData(&d_typePermutation , "permutation", "Type of fill reducing permutation"))
    , d_L_nnz(initData(&d_L_nnz, 0, "L_nnz", "Number of non-zero values in the lower triangular matrix of the factorization. The lower, the faster the system is solved.", true, true))
{
    sofa::helper::OptionsGroup d_typePermutationOptions{"None", "SuiteSparse", "METIS"};
    d_typePermutationOptions.setSelectedItem(0); // default None
    d_typePermutation.setValue(d_typePermutationOptions);
}


template<class TMatrix, class TVector,class TThreadManager>
void SparseLUSolver<TMatrix,TVector,TThreadManager>::solve (Matrix& M, Vector& x, Vector& b)
{
    SparseLUInvertData<Real> * invertData = (SparseLUInvertData<Real>*) this->getMatrixInvertData(&M);
    int n = invertData->A.n;

    {
        sofa::helper::ScopedAdvancedTimer solveTimer("solve");
        switch( d_typePermutation.getValue().getSelectedId() )
        {
            
            case 0://None->Identity

            {   
                cs_ipvec (n, nullptr,  b.ptr(), x.ptr()) ;	// copy
                cs_lsolve (invertData->N->L, x.ptr() ) ;		// x = L\x 
                cs_usolve (invertData->N->U, x.ptr() ) ;		// x = U\x 
                break;
            }

            case 1://SuiteSparse
            {
                cs_ipvec (n, invertData->N->Pinv, b.ptr(), invertData->tmp) ;	// x = P*b , partial pivot
                cs_lsolve (invertData->N->L, invertData->tmp) ;		// x = L\x 
                cs_usolve (invertData->N->U, invertData->tmp) ;		// x = U\x 
                cs_ipvec (n, invertData->S->Q, invertData->tmp, x.ptr()) ;	// x = Q*x fill reducing permutation on columns only
                break;
            }
            
            case 2://METIS
            {
                // no partial pivoting
                cs_pvec (n, invertData->perm.data() , b.ptr(), invertData->tmp) ; // x = P*b permutation on rows
                cs_lsolve (invertData->N->L, invertData->tmp) ;		// x = L\x
                cs_usolve (invertData->N->U, invertData->tmp) ;		// x = U\x
                cs_pvec (n, invertData->iperm.data() , invertData->tmp , x.ptr()) ;	// x = Q*x permutation on columns
                break;
            }

            default:
                break;
        }
    }
}

template<class TMatrix, class TVector,class TThreadManager>
void SparseLUSolver<TMatrix,TVector,TThreadManager>::invert(Matrix& M)
{
    SparseLUInvertData<Real> * invertData = (SparseLUInvertData<Real>*) this->getMatrixInvertData(&M);

    sofa::linearalgebra::CompressedRowSparseMatrix<Real>* matrix;

    if constexpr (!std::is_same_v<Matrix, decltype(Mfiltered)>)
    {
        if (!Mfiltered)
        {
            Mfiltered = std::make_unique<sofa::linearalgebra::CompressedRowSparseMatrix<Real> >();
        }
        Mfiltered->copyNonZeros(M);
        Mfiltered->compress();

        matrix = Mfiltered.get();
    }
    else
    {
        M.compress();
        matrix = &M;
    }

    if (invertData->N) cs_nfree(invertData->N);
    if (invertData->tmp) cs_free(invertData->tmp);

    //build A with M
    invertData->A.nzmax = matrix->getColsValue().size();	// maximum number of entries
    invertData->A.m = matrix->rowSize();					// number of rows
    invertData->A.n = matrix->colSize();					// number of columns
    invertData->A_p = matrix->getRowBegin();
    invertData->A.p = (int *) &(invertData->A_p[0]);							// column pointers (size n+1) or col indices (size nzmax)
    invertData->A_i = matrix->getColsIndex();
    invertData->A.i = (int *) &(invertData->A_i[0]);							// row indices, size nzmax
    invertData->A_x = matrix->getColsValue();
    invertData->A.x = (Real *) &(invertData->A_x[0]);				// numerical values, size nzmax
    invertData->A.nz = -1;							// # of entries in triplet matrix, -1 for compressed-col
    cs_dropzeros( &invertData->A );

    invertData->notSameShape = compareMatrixShape(invertData->A.n , (int*) invertData->A_p.data() , (int*) invertData->A_i.data(), (invertData->Previous_colptr.size())-1 ,invertData->Previous_colptr.data() ,invertData->Previous_rowind.data() );

    invertData->tmp = (Real *) cs_malloc (invertData->A.n, sizeof (Real)) ;

    {
        sofa::helper::ScopedAdvancedTimer factorizationTimer("factorization");
        switch( d_typePermutation.getValue().getSelectedId() )
        {
            case 0://None->Identity
            {  
                if (invertData->S) 
                {
                    cs_sfree(invertData->S);
                }
                invertData->S = symbolic_LU( &(invertData->A) ) ;	// ordering and symbolic analysis 
                invertData->N = cs_lu (&invertData->A, invertData->S, f_tol.getValue()) ;		// numeric LU factorization 
                break;
            }

            case 1://SuiteSparse
            {
                if (invertData->notSameShape ) 
                {
                    if (invertData->S) 
                    {
                        cs_sfree(invertData->S);
                    }
                    invertData->S = cs_sqr (&invertData->A, 1 , 0) ; // Suitsparse compute fill reducing permutation for LU decomposition
                }
                invertData->N = cs_lu (&invertData->A, invertData->S, f_tol.getValue()) ;		// numeric LU factorization 
                break;
            }

            case 2://Metis
            {
                if (invertData->notSameShape )
                {
                    invertData->perm.resize(invertData->A.n);
                    invertData->iperm.resize(invertData->A.n);

                    fillReducingPermutation(invertData->A, invertData->perm.data(), invertData->iperm.data() ); //METIS compute the fill reducing permutation
                }

                invertData->permuted_A = cs_permute(&(invertData->A), invertData->iperm.data(), invertData->perm.data(), 1);

                if (invertData->notSameShape ) 
                {
                    if (invertData->S) 
                    {
                        cs_sfree(invertData->S);
                    }
                    invertData->S = symbolic_LU( invertData->permuted_A );
                }
                
                invertData->N = cs_lu ( invertData->permuted_A, invertData->S, f_tol.getValue()) ;		// numeric LU factorization 

                cs_free(invertData->permuted_A); // prevent memory leak

                break;
            }

            default:
                break;
        }
    }

    d_L_nnz.setValue(invertData->N->L->nzmax);

    // store the shape of the matrix
    if ( invertData->notSameShape )
    {
        invertData->Previous_rowind.clear();
        invertData->Previous_colptr.resize( (invertData->A.n) +1);

        for (int i=0 ; i<invertData->A.n ; i++)
        {
            invertData->Previous_colptr[i+1] = invertData->A_p[i+1];

            for ( int j = (int) invertData->A_p[i] ; j < (int)invertData->A_p[i+1] ; j++)
            {
                invertData->Previous_rowind.push_back( invertData->A_i[j]);
            }
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
    S->Q = nullptr; // should have been the fill permutation computed by SuiteSparse, not used here
    return S ;
}

} // namespace sofa::component::linearsolver::direct
