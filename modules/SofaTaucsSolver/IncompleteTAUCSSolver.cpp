/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
// Author: Hadrien Courtecuisse
//
// Copyright: See COPYING file that comes with this distribution
#include <SofaTaucsSolver/IncompleteTAUCSSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <math.h>
#include <sofa/helper/system/thread/CTime.h>
#include <SofaBaseLinearSolver/MatrixLinearSolver.cpp>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.inl>

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
    :
#ifdef VAIDYA
    f_incompleteType( initData(&f_incompleteType,0,"incompleteType","0 = Incomplete Cholesky (ic), 1 = vaidya (va) , 2 = recursive vaidya (vr) ") ) ,
#endif
    f_ordering( initData(&f_ordering,2,"ordering","ose ordering 0=identity/1=tree/2=metis/3=natural/4=genmmd/5=md/6=mmd/7=amd") )
    , f_dropTol( initData(&f_dropTol,(double) 0.0,"ic_dropTol","Drop tolerance use for incomplete factorization") )
    , f_modified_flag( initData(&f_modified_flag,true,"ic_modifiedFlag","Modified ICC : maintains rowsums") )
#ifdef VAIDYA
    , f_subgraphs( initData(&f_subgraphs,(double) 10,"va_subgraphs","Desired number of subgraphs for Vaidya") )
    , f_stretch_flag( initData(&f_stretch_flag,false,"va_stretch_flag","Starting with a low stretch tree") )
    , f_multifrontal( initData(&f_multifrontal,true,"va_multifrontal","Use multifrontal algo in vaidya") )
    , f_seed( initData(&f_seed,123,"va_seed","Affects decomposition to subtrees") )
    , f_C( initData(&f_C,0.25,"vr_C","splits tree into about k=f_C*n^(1/(1+f_epsilon)) subgraphs") )
    , f_epsilon( initData(&f_epsilon,(double) 0.2,"vr_epsilon","splits tree into about k=f_C*n^(1/(1+f_epsilon)) subgraphs") )
    , f_nsmall( initData(&f_nsmall,10000,"vr_nsmall","matrices smaller than nsmall are factored directly") )
    , f_maxlevels( initData(&f_maxlevels,2,"vr_maxlevels","preconditioner has at most l levels") )
    , f_innerits( initData(&f_innerits,2,"vr_innerits","using at most m iterations") )
    , f_innerconv( initData(&f_innerconv,(double) 0.01,"vr_innerconv","inner solves reduce their residual by a factor of r") )
#endif
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

    IncompleteTAUCSSolverInvertData * data = (IncompleteTAUCSSolverInvertData *) getMatrixInvertData(&M);

    if (data->perm) free(data->perm);
    if (data->invperm) free(data->invperm);

    data->Mfiltered.copyUpperNonZeros(M);
    data->Mfiltered.fullRows();

    data->matrix_taucs.n = data->Mfiltered.rowSize();
    data->matrix_taucs.m = data->Mfiltered.colSize();
    data->matrix_taucs.flags = get_taucs_incomplete_flags<Real>() | TAUCS_SYMMETRIC | TAUCS_LOWER; // Upper on row-major is actually lower on column-major transposed matrix
    data->matrix_taucs.colptr = (int *) &(data->Mfiltered.getRowBegin()[0]);
    data->matrix_taucs.rowind = (int *) &(data->Mfiltered.getColsIndex()[0]);
    data->matrix_taucs.values.d = (double*) &(data->Mfiltered.getColsValue()[0]);

    data->B.resize(data->matrix_taucs.n);
    data->R.resize(data->matrix_taucs.n);

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

#ifdef VAIDYA
    if (f_incompleteType.getValue()==0)
    {
#endif
        data->freeL();

        taucs_ccs_order(&data->matrix_taucs,&data->perm,&data->invperm,ordering);
        taucs_ccs_matrix* PAPT = taucs_ccs_permute_symmetrically(&data->matrix_taucs,data->perm,data->invperm);
        data->L = taucs_ccs_factor_llt(PAPT,f_dropTol.getValue(),f_modified_flag.getValue());
        taucs_ccs_free(PAPT);

        data->precond_fn = taucs_ccs_solve_llt;
        data->precond_args = data->L;

#ifdef VAIDYA
    }
    else if (f_incompleteType.getValue()==1)
    {
        data->freeL();

        srand(f_seed.getValue());
        int rnd = rand();

        taucs_ccs_matrix *  V = taucs_amwb_preconditioner_create(&data->matrix_taucs,rnd,f_subgraphs.getValue(),f_stretch_flag.getValue(),0);
        taucs_ccs_order(V,&data->perm,&data->invperm,ordering);
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

        data->precond_fn = taucs_ccs_solve_llt;
        data->precond_args = data->L;
    }
    else if (f_incompleteType.getValue()==2)
    {
        data->freeRL();

        data->RL = (recvaidya_args *) taucs_recursive_amwb_preconditioner_create(&data->matrix_taucs,
                f_C.getValue(),
                f_epsilon.getValue(),
                f_nsmall.getValue(),
                f_maxlevels.getValue(),
                f_innerits.getValue(),
                f_innerconv.getValue(),
                &data->perm,
                &data->invperm);
        data->precond_fn   = taucs_recursive_amwb_preconditioner_solve;
        data->precond_args = data->RL;
    }
#endif

    taucs_logfile((char*)"none");
}

template<class TMatrix, class TVector>
void IncompleteTAUCSSolver<TMatrix,TVector>::solve (Matrix& M, Vector& z, Vector& r)
{
    IncompleteTAUCSSolverInvertData * data = (IncompleteTAUCSSolverInvertData *) getMatrixInvertData(&M);

    for (int i=0; i<data->matrix_taucs.n; i++) data->B[i] = r[data->perm[i]];
    data->precond_fn(data->precond_args,&data->R[0],&data->B[0]);
    for (int i=0; i<data->matrix_taucs.n; i++) z[i] = data->R[data->invperm[i]];
}


SOFA_DECL_CLASS(IncompleteTAUCSSolver)

int IncompleteTAUCSSolverClass = core::RegisterObject("Direct linear solvers implemented with the TAUCS library")
        .add< IncompleteTAUCSSolver< CompressedRowSparseMatrix<double>,FullVector<double> > >()
        .add< IncompleteTAUCSSolver< CompressedRowSparseMatrix<defaulttype::Mat<3,3,double> >,FullVector<double> > >(true)
        ;

} // namespace linearsolver

} // namespace component

} // namespace sofa

