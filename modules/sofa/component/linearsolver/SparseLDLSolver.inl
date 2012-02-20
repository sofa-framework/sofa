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

#ifndef SOFA_COMPONENT_LINEARSOLVER_SparseLDLSolver_INL
#define SOFA_COMPONENT_LINEARSOLVER_SparseLDLSolver_INL

#include <sofa/component/linearsolver/SparseLDLSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <math.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.inl>

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
SparseLDLSolver<TMatrix,TVector>::SparseLDLSolver()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
{
}

template<class TMatrix, class TVector>
void SparseLDLSolver<TMatrix,TVector>::solve (Matrix& M, Vector& z, Vector& r)
{
    SparseLDLSolverInvertData * data = (SparseLDLSolverInvertData *) getMatrixInvertData(&M);

    // permutation according to metis
    for (int i=0; i<data->n; i++) B[i] = r[data->perm[i]];

    for (int j = 0 ; j < data->n ; j++)
    {
        for (int p = data->colptr [j] ; p < data->colptr[j+1] ; p++)
        {
            B [data->rowind [p]] -= data->values[p] * B[j] ;
        }
    }
    for (int j = 0 ; j < data->n ; j++)
    {
        B [j] /= data->D[j] ;
    }
    for (int j = data->n-1 ; j >= 0 ; j--)
    {
        for (int p = data->colptr[j] ; p < data->colptr[j+1] ; p++)
        {
            B [j] -= data->values[p] * B [data->rowind[p]] ;
        }
    }

    for (int i=0; i<data->n; i++) z[i] = B[data->invperm[i]];
}

template<class TMatrix, class TVector>
void SparseLDLSolver<TMatrix,TVector>::invert(Matrix& M)
{
    SparseLDLSolverInvertData * data = (SparseLDLSolverInvertData *) getMatrixInvertData(&M);

    //remplir A avec M
    data->n = M.colSize();// number of columns
    data->Mfiltered.clear();
    data->Mfiltered.resize(M.rowSize(),M.colSize());
    data->Mfiltered.copyNonZeros(M);
    data->Mfiltered.compress();

    int * Mcolptr = (int *) &data->Mfiltered.getRowBegin()[0];
    int * Mrowind = (int *) &data->Mfiltered.getColsIndex()[0];
    Real * Mvalues = (Real *) &data->Mfiltered.getColsValue()[0];

    data->perm.resize(data->n);
    data->invperm.resize(data->n);
    B.resize(data->n);
    data->D.resize(data->n);

    LDL_ordering(data->n,Mcolptr,Mrowind,&data->perm[0],&data->invperm[0]);
    LDL_symbolic(data->n,Mcolptr,Mrowind,&data->colptr[0],&data->perm[0],&data->invperm[0]);

    data->nnz = data->colptr[data->n];
    data->rowind.resize(data->nnz);
    data->values.resize(data->nnz);

    LDL_numeric(data->n,Mcolptr,Mrowind,Mvalues,&data->colptr[0],&data->rowind[0],&data->values[0],&data->D[0],&data->perm[0],&data->invperm[0]);
}

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
