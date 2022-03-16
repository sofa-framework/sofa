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

namespace sofa::component::linearsolver::direct
{

template<class TMatrix, class TVector>
SparseCholeskySolver<TMatrix,TVector>::SparseCholeskySolver()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , S(nullptr), N(nullptr)
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

    cs_ipvec (n, S->Pinv, (double*) &(r_tmp[0]), (double*) &(tmp[0]));	//x = P*b

    cs_lsolve (N->L, (double*) &(tmp[0]));			//x = L\x
    cs_ltsolve (N->L, (double*) &(tmp[0]));			//x = L'\x/

    cs_pvec (n, S->Pinv, (double*) &(tmp[0]), (double*) &(z_tmp[0]));	 //b = P'*x

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
    int order = -1; //?????
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
    S = cs_schol (&A, order) ;		/* ordering and symbolic analysis */
    N = cs_chol (&A, S) ;		/* numeric Cholesky factorization */
}


} // namespace sofa::component::linearsolver::direct
