/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

template<class TMatrix, class TVector,class TThreadManager>
SparseLUSolver<TMatrix,TVector,TThreadManager>::SparseLUSolver()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , f_tol( initData(&f_tol,0.001,"tolerance","tolerance of factorization") )
{
}


template<class TMatrix, class TVector,class TThreadManager>
void SparseLUSolver<TMatrix,TVector,TThreadManager>::solve (Matrix& M, Vector& z, Vector& r)
{
    SparseLUInvertData<Real> * invertData = (SparseLUInvertData<Real>*) this->getMatrixInvertData(&M);
    int n = invertData->A.n;

    cs_ipvec (n, invertData->N->Pinv, r.ptr(), invertData->tmp) ;	/* x = P*b */
    cs_lsolve (invertData->N->L, invertData->tmp) ;		/* x = L\x */
    cs_usolve (invertData->N->U, invertData->tmp) ;		/* x = U\x */
    cs_ipvec (n, invertData->S->Q, invertData->tmp, z.ptr()) ;	/* b = Q*x */
}

template<class TMatrix, class TVector,class TThreadManager>
void SparseLUSolver<TMatrix,TVector,TThreadManager>::invert(Matrix& M)
{
    SparseLUInvertData<Real> * invertData = (SparseLUInvertData<Real>*) this->getMatrixInvertData(&M);
    int order = -1; //?????

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

    //M.check_matrix();
    //CompressedRowSparseMatrix<double>::check_matrix(-1 /*A.nzmax*/,A.m,A.n,A.p,A.i,A.x);
    //sout << "diag =";
    //for (int i=0;i<A.n;++i) sout << " " << M.element(i,i);
    //sout << sendl;
    //sout << "SparseCholeskySolver: start factorization, n = " << A.n << " nnz = " << A.p[A.n] << sendl;
    invertData->tmp = (Real *) cs_malloc (invertData->A.n, sizeof (Real)) ;
    invertData->S = cs_sqr (&invertData->A, order, 0) ;		/* ordering and symbolic analysis */
    invertData->N = cs_lu (&invertData->A, invertData->S, f_tol.getValue()) ;		/* numeric LU factorization */
    //sout << "SparseCholeskySolver: factorization complete, nnz = " << N->L->p[N->L->n] << sendl;
}


} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
