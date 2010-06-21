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
    : f_incompleteType( initData(&f_incompleteType,0,"incompleteType","0 = Incomplete Cholesky, 1 = vaidya") )
    , f_ordering( initData(&f_ordering,2,"ordering","ose ordering 0=identity/1=tree/2=metis/3=natural/4=genmmd/5=md/6=mmd/7=amd") )
    , f_dropTol( initData(&f_dropTol,0.0,"ic_dropTol","Drop tolerance use for incomplete factorization") )
    , f_modified_flag( initData(&f_modified_flag,false,"ic_modifiedFlag","Modified ICC : maintains rowsums") )
    , f_subgraphs( initData(&f_subgraphs,1.0,"va_subgraphs","Desired number of subgraphs for Vaidya") )
    , f_stretch_flag( initData(&f_stretch_flag,false,"va_stretch_flag","Modified ICC : maintains rowsums") )
    , f_multifrontal( initData(&f_multifrontal,false,"va_multifrontal","Use multifrontal algo in vaidya") )
    , f_seed( initData(&f_seed,123,"va_seed","Affects decomposition to subtrees") )
{
    new_perm = false;
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
    if (data->L) taucs_ccs_free(data->L);

    data->Mfiltered.copyUpperNonZeros(M);
    data->Mfiltered.fullRows();

    data->matrix_taucs.n = data->Mfiltered.rowSize();
    data->matrix_taucs.m = data->Mfiltered.colSize();
    data->matrix_taucs.flags = get_taucs_incomplete_flags<Real>() | TAUCS_SYMMETRIC | TAUCS_LOWER; // Upper on row-major is actually lower on column-major transposed matrix
    data->matrix_taucs.colptr = (int *) &(data->Mfiltered.getRowBegin()[0]);
    data->matrix_taucs.rowind = (int *) &(data->Mfiltered.getColsIndex()[0]);
    data->matrix_taucs.values.d = (double*) &(data->Mfiltered.getColsValue()[0]);

    if (this->f_printLog.getValue()) taucs_logfile((char*)"stdout");

    char * ordering;
    switch (f_ordering.getValue())
    {
    case 0  : ordering = (char *) "identity"; break;
    case 1  : ordering = (char *) "tree"; break;
    case 2  : ordering = (char *) "metis"; break;
    case 3  : ordering = (char *) "natural"; break;
    case 4  : ordering = (char *) "genmmd"; break;
    case 5  : ordering = (char *) "md"; break;
    case 6  : ordering = (char *) "mmd"; break;
    case 7  : ordering = (char *) "amd"; break;
    default : ordering = (char *) "identity"; break;
    }

    if (f_incompleteType.getValue()==0)
    {
        taucs_ccs_order(&data->matrix_taucs,&data->perm,&data->invperm,ordering);
        taucs_ccs_matrix* PAPT = taucs_ccs_permute_symmetrically(&data->matrix_taucs,data->perm,data->invperm);
        data->L = taucs_ccs_factor_llt(PAPT,f_dropTol.getValue(),(int) f_modified_flag.getValue());

        taucs_ccs_free(PAPT);
    }
    else if (f_incompleteType.getValue()==1)
    {
        srand(f_seed.getValue());
        int rnd = rand();

        taucs_ccs_matrix *  V = taucs_amwb_preconditioner_create(&data->matrix_taucs,rnd,f_subgraphs.getValue(),f_stretch_flag.getValue(),0);
        taucs_ccs_order(V,&data->perm,&data->invperm,ordering);

        //data->PAPT = taucs_ccs_permute_symmetrically(&data->matrix_taucs,data->perm,data->invperm);
        taucs_ccs_matrix *  PVPT = taucs_ccs_permute_symmetrically(V,data->perm,data->invperm);

        taucs_ccs_free(V);

        if (f_multifrontal.getValue())
        {
            void* snL = taucs_ccs_factor_llt_mf(PVPT);
            data->L = taucs_supernodal_factor_to_ccs(snL);
            taucs_supernodal_factor_free(snL);
        }
        else
        {
            data->L = taucs_ccs_factor_llt(PVPT,0.0,0);
        }

        taucs_ccs_free(PVPT);

        data->L->flags |= TAUCS_DOUBLE;
    }

    taucs_logfile((char*)"none");

    data->B.resize(data->L->n);
    data->R.resize(data->L->n);
    new_perm = true;
}

template<class TMatrix, class TVector>
void IncompleteTAUCSSolver<TMatrix,TVector>::solve (Matrix& M, Vector& z, Vector& r)
{
    IncompleteTAUCSSolverInvertData * data = (IncompleteTAUCSSolverInvertData *) M.getMatrixInvertData();
    if (data==NULL)
    {
        z = r;
        std::cerr << "Error the matrix is not factorized" << std::endl;
        return;
    }

    if (new_perm) for (int i=0; i<data->L->n; i++) data->B[i] = r[data->perm[i]];
    taucs_ccs_solve_llt(data->L,&data->R[0],&data->B[0]);
    for (int i=0; i<data->L->n; i++) z[i] = data->R[data->invperm[i]];
}


SOFA_DECL_CLASS(IncompleteTAUCSSolver)

int IncompleteTAUCSSolverClass = core::RegisterObject("Direct linear solvers implemented with the TAUCS library")
        .add< IncompleteTAUCSSolver< CompressedRowSparseMatrix<double>,FullVector<double> > >(true)
        ;

} // namespace linearsolver

} // namespace component

} // namespace sofa

