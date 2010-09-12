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
#include <sofa/component/linearsolver/SparseTAUCSLUSolver.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <math.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/component/linearsolver/ParallelMatrixLinearSolver.inl>
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
SparseTAUCSLUSolver<TMatrix,TVector>::SparseTAUCSLUSolver()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , f_dropTol( initData(&f_dropTol,(double) 0.0,"ic_dropTol","Drop tolerance use for incomplete factorization") )
    , f_nproc_simu( initData(&f_nproc_simu,(unsigned) 1,"nproc_simu","NB proc used for the simulation") )
#ifdef SOFA_HAVE_CILK
    , f_nproc_fact( initData(&f_nproc_fact,(unsigned) 1,"nproc_fact","NB proc used for the factorization") )
#endif
{
}

template<class T>
int get_taucs_lu_flags();

template<>
int get_taucs_lu_flags<double>() { return TAUCS_DOUBLE; }

template<>
int get_taucs_lu_flags<float>() { return TAUCS_SINGLE; }

template<class TMatrix, class TVector>
void SparseTAUCSLUSolver<TMatrix,TVector>::invert(Matrix& M)
{
    M.compress();

    SparseTAUCSLUSolverInvertData * data = (SparseTAUCSLUSolverInvertData *) M.getMatrixInvertData();
    if (data==NULL)
    {
        M.setMatrixInvertData(new SparseTAUCSLUSolverInvertData());
        data = (SparseTAUCSLUSolverInvertData *) M.getMatrixInvertData();
    }

    if (data->perm) free(data->perm);
    if (data->invperm) free(data->invperm);
    if (data->PAPT) taucs_ccs_free(data->PAPT);

    data->Mfiltered.copyUpperNonZeros(M);
    data->Mfiltered.fullRows();

    data->matrix_taucs.n = data->Mfiltered.rowSize();
    data->matrix_taucs.m = data->Mfiltered.colSize();
    data->matrix_taucs.flags = get_taucs_lu_flags<Real>() | TAUCS_SYMMETRIC | TAUCS_LOWER; // Upper on row-major is actually lower on column-major transposed matrix
    data->matrix_taucs.colptr = (int *) &(data->Mfiltered.getRowBegin()[0]);
    data->matrix_taucs.rowind = (int *) &(data->Mfiltered.getColsIndex()[0]);
    data->matrix_taucs.values.d = (double*) &(data->Mfiltered.getColsValue()[0]);

    if (this->f_printLog.getValue()) taucs_logfile((char*)"stdout");

    taucs_ccs_order(&data->matrix_taucs,&data->perm,&data->invperm,(char *) "metis");
    data->PAPT = taucs_ccs_permute_symmetrically(&data->matrix_taucs,data->perm,data->invperm);
//     if (f_dropTol.getValue()) {
    data->L = taucs_ccs_factor_llt(data->PAPT,f_dropTol.getValue(),true);
//     } else {
//       data->L = (taucs_ccs_matrix*) taucs_ccs_factor_llt_ll_maxdepth(data->PAPT,0.0);
//     }

    taucs_logfile((char*)"none");

    data->B.resize(data->matrix_taucs.n);
}

template<class TMatrix, class TVector>
void SparseTAUCSLUSolver<TMatrix,TVector>::solve (Matrix& M, Vector& z, Vector& r)
{
    SparseTAUCSLUSolverInvertData * data = (SparseTAUCSLUSolverInvertData *) M.getMatrixInvertData();
    if (data==NULL)
    {
        z = r;
        return;
    }

    // permutation according to metis
    for (int i=0; i<data->matrix_taucs.n; i++) data->B[i] = r[data->perm[i]];

    //L is strore as L^T
    //first solve L B = r
    for (int row=0; row<data->L->n; row++)
    {
        int begin = data->L->colptr[row];
        int end   = data->L->colptr[row+1];

        int diag = data->L->rowind[begin];
        data->B[diag] /= data->L->values.d[begin];

        for (int i=begin+1; i<end; i++)
        {
            int col = data->L->rowind[i];
            double val = data->L->values.d[i];

            data->B[col] -= val * data->B[diag];
        }
    }

    //next solve L^T R = B
    for (int row=data->L->n; row>0; row--)
    {
        int begin = data->L->colptr[row-1];
        int end   = data->L->colptr[row];

        int diag = data->L->rowind[begin];

        for (int i=begin+1; i<end; i++)
        {
            int col = data->L->rowind[i];
            double val = data->L->values.d[i];

            data->B[diag] -= val * data->B[col];
        }

        data->B[diag] /= data->L->values.d[begin];
    }
    for (int i=0; i<data->matrix_taucs.n; i++) z[i] = data->B[data->invperm[i]];
}


SOFA_DECL_CLASS(SparseTAUCSLUSolver)

int SparseTAUCSLUSolverClass = core::RegisterObject("Direct linear solvers implemented with the TAUCS library")
        .add< SparseTAUCSLUSolver< CompressedRowSparseMatrix<double>,FullVector<double> > >()
        .add< SparseTAUCSLUSolver< CompressedRowSparseMatrix<defaulttype::Mat<3,3,double> >,FullVector<double> > >(true)
        ;

} // namespace linearsolver

} // namespace component

} // namespace sofa

