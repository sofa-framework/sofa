/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
// Author: Hadrien Courtecuisse
//
// Copyright: See COPYING file that comes with this distribution
#include <SofaSparseSolver/SparseLUSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <math.h>
#include <sofa/helper/system/thread/CTime.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>

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

template<class TMatrix, class TVector>
SparseLUSolver<TMatrix,TVector>::SparseLUSolver()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , f_tol( initData(&f_tol,0.001,"tolerance","tolerance of factorization") )
    , S(NULL), N(NULL), tmp(NULL)
{
}

template<class TMatrix, class TVector>
SparseLUSolver<TMatrix,TVector>::~SparseLUSolver()
{
    if (S) cs_sfree (S);
    if (N) cs_nfree (N);
    if (tmp) cs_free (tmp);
}


template<class TMatrix, class TVector>
void SparseLUSolver<TMatrix,TVector>::solve (Matrix& /*M*/, Vector& z, Vector& r)
{
    int n = A.n;

    cs_ipvec (n, N->Pinv, r.ptr(), tmp) ;	/* x = P*b */
    cs_lsolve (N->L, tmp) ;		/* x = L\x */
    cs_usolve (N->U, tmp) ;		/* x = U\x */
    cs_ipvec (n, S->Q, tmp, z.ptr()) ;	/* b = Q*x */
}

template<class TMatrix, class TVector>
void SparseLUSolver<TMatrix,TVector>::invert(Matrix& M)
{
    int order = -1; //?????

    if (S) cs_sfree(S);
    if (N) cs_nfree(N);
    if (tmp) cs_free(tmp);
    M.compress();
    //remplir A avec M
    A.nzmax = M.getColsValue().size();	// maximum number of entries
    A.m = M.rowBSize();					// number of rows
    A.n = M.colBSize();					// number of columns
    A_p = M.getRowBegin();
    A.p = (int *) &(A_p[0]);							// column pointers (size n+1) or col indices (size nzmax)
    A_i = M.getColsIndex();
    A.i = (int *) &(A_i[0]);							// row indices, size nzmax
    A_x = M.getColsValue();
    A.x = (double *) &(A_x[0]);				// numerical values, size nzmax
    A.nz = -1;							// # of entries in triplet matrix, -1 for compressed-col
    cs_dropzeros( &A );

    //M.check_matrix();
    //CompressedRowSparseMatrix<double>::check_matrix(-1 /*A.nzmax*/,A.m,A.n,A.p,A.i,A.x);
    //sout << "diag =";
    //for (int i=0;i<A.n;++i) sout << " " << M.element(i,i);
    //sout << sendl;
    //sout << "SparseCholeskySolver: start factorization, n = " << A.n << " nnz = " << A.p[A.n] << sendl;
    tmp = (double *) cs_malloc (A.n, sizeof (double)) ;
    S = cs_sqr (&A, order, 0) ;		/* ordering and symbolic analysis */
    N = cs_lu (&A, S, f_tol.getValue()) ;		/* numeric LU factorization */
    //sout << "SparseCholeskySolver: factorization complete, nnz = " << N->L->p[N->L->n] << sendl;
}

SOFA_DECL_CLASS(SparseLUSolver)

int SparseLUSolverClass = core::RegisterObject("Direct linear solver based on Sparse LU factorization, implemented with the CSPARSE library")
        .add< SparseLUSolver< CompressedRowSparseMatrix<double>,FullVector<double> > >(true)
        ;

} // namespace linearsolver

} // namespace component

} // namespace sofa

