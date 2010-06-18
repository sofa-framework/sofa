/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/component/linearsolver/IncompleteTAUCSSolver.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <math.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.h>

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
IncompleteTAUCSSolver<TMatrix,TVector>::IncompleteTAUCSSolver()
    : f_options( initData(&f_options,"options","TAUCS unified solver list of space-separated options") )
    , f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , f_use_metis( initData(&f_use_metis,true,"use_metis","Use metis to reorder the matrix") )
    , f_dropFactorTol( initData(&f_dropFactorTol,0.0,"dropFactorTol","Drop tolerance use for incomplete factorization") )
    , f_modified_flag( initData(&f_modified_flag,false,"modified_flag","Factoring will be Modified ICC") )
{
}

template<class T>
int get_taucs_incomplete_flags();

template<>
int get_taucs_incomplete_flags<double>() { return TAUCS_DOUBLE; }

template<>
int get_taucs_incomplete_flags<float>() { return TAUCS_SINGLE; }

template<class TMatrix, class TVector>
void IncompleteTAUCSSolver<TMatrix,TVector>::invert(Matrix& M)
{
    M.compress();

    IncompleteTAUCSSolverInvertData * data = (IncompleteTAUCSSolverInvertData *) M.getMatrixInvertData();
    if (data==NULL)
    {
        M.setMatrixInvertData(new IncompleteTAUCSSolverInvertData());
        data = (IncompleteTAUCSSolverInvertData *) M.getMatrixInvertData();
    }

    if (data->perm) free(data->perm);
    if (data->invperm) free(data->invperm);
    if (data->PAPT) taucs_ccs_free(data->PAPT);
    if (data->L) taucs_ccs_free(data->L);

    int modified_flag = (int) f_modified_flag.getValue();

    data->Mfiltered.copyUpperNonZeros(M);
    data->Mfiltered.fullRows();

    data->matrix_taucs.n = data->Mfiltered.rowSize();
    data->matrix_taucs.m = data->Mfiltered.colSize();
    data->matrix_taucs.flags = get_taucs_incomplete_flags<Real>();
    data->matrix_taucs.flags |= TAUCS_SYMMETRIC;
    data->matrix_taucs.flags |= TAUCS_LOWER; // Upper on row-major is actually lower on column-major transposed matrix
    data->matrix_taucs.colptr = (int *) &(data->Mfiltered.getRowBegin()[0]);
    data->matrix_taucs.rowind = (int *) &(data->Mfiltered.getColsIndex()[0]);
    data->matrix_taucs.values.d = (double*) &(data->Mfiltered.getColsValue()[0]);

    char* ordering;
    if (f_use_metis.getValue()) ordering = (char *) "metis";
    else ordering = (char *) "identity";

    if (this->f_printLog.getValue()) taucs_logfile((char*)"stdout");

    taucs_ccs_order(&data->matrix_taucs,&data->perm,&data->invperm,ordering);
    data->PAPT = taucs_ccs_permute_symmetrically(&data->matrix_taucs,data->perm,data->invperm);
    data->L = taucs_ccs_factor_llt(data->PAPT,f_dropFactorTol.getValue(),modified_flag);

    taucs_logfile((char*)"none");
}

template<class TMatrix, class TVector>
void IncompleteTAUCSSolver<TMatrix,TVector>::solve (Matrix& M, Vector& z, Vector& r)
{
    IncompleteTAUCSSolverInvertData * data = (IncompleteTAUCSSolverInvertData *) M.getMatrixInvertData();
    if (data==NULL || data->L==NULL)
    {
        z = r;
        std::cerr << "Error the matrix is not factorized" << std::endl;
        return;
    }

    taucs_ccs_solve_llt(data->L,&z[0],&r[0]);
}


SOFA_DECL_CLASS(IncompleteTAUCSSolver)

int IncompleteTAUCSSolverClass = core::RegisterObject("Direct linear solvers implemented with the TAUCS library")
        .add< IncompleteTAUCSSolver< CompressedRowSparseMatrix<double>,FullVector<double> > >(true)
        ;

} // namespace linearsolver

} // namespace component

} // namespace sofa

