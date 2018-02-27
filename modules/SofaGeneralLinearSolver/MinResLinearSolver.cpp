/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
// Author: Matthieu Nesme, INRIA, (C) 2012
#define SOFA_COMPONENT_LINEARSOLVER_MINRESLINEARSOLVER_CPP
#include <SofaGeneralLinearSolver/MinResLinearSolver.inl>

#include <sofa/core/ObjectFactory.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace linearsolver
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(MinResLinearSolver)

int MinResLinearSolverClass = core::RegisterObject("Linear system solver using the MINRES iterative algorithm")
        .add< MinResLinearSolver< GraphScatteredMatrix, GraphScatteredVector > >(true)
#ifndef SOFA_FLOAT
        .add< MinResLinearSolver< FullMatrix<double>, FullVector<double> > >()
        .add< MinResLinearSolver< SparseMatrix<double>, FullVector<double> > >()
        .add< MinResLinearSolver< CompressedRowSparseMatrix<double>, FullVector<double> > >()
        .add< MinResLinearSolver< CompressedRowSparseMatrix<Mat<2,2,double> >, FullVector<double> > >()
        .add< MinResLinearSolver< CompressedRowSparseMatrix<Mat<3,3,double> >, FullVector<double> > >()
        .add< MinResLinearSolver< CompressedRowSparseMatrix<Mat<4,4,double> >, FullVector<double> > >()
        .add< MinResLinearSolver< CompressedRowSparseMatrix<Mat<6,6,double> >, FullVector<double> > >()
        .add< MinResLinearSolver< CompressedRowSparseMatrix<Mat<8,8,double> >, FullVector<double> > >()
#endif
#ifndef SOFA_DOUBLE
        .add< MinResLinearSolver< CompressedRowSparseMatrix<float>, FullVector<float> > >()
        .add< MinResLinearSolver< CompressedRowSparseMatrix<Mat<2,2,float> >, FullVector<float> > >()
        .add< MinResLinearSolver< CompressedRowSparseMatrix<Mat<3,3,float> >, FullVector<float> > >()
        .add< MinResLinearSolver< CompressedRowSparseMatrix<Mat<4,4,float> >, FullVector<float> > >()
        .add< MinResLinearSolver< CompressedRowSparseMatrix<Mat<6,6,float> >, FullVector<float> > >()
        .add< MinResLinearSolver< CompressedRowSparseMatrix<Mat<8,8,float> >, FullVector<float> > >()
#endif
        .addAlias("MINRESSolver")
        .addAlias("MinResSolver")
        ;

template class SOFA_GENERAL_LINEAR_SOLVER_API MinResLinearSolver< GraphScatteredMatrix, GraphScatteredVector >;
#ifndef SOFA_FLOAT
template class SOFA_GENERAL_LINEAR_SOLVER_API MinResLinearSolver< FullMatrix<double>, FullVector<double> >;
template class SOFA_GENERAL_LINEAR_SOLVER_API MinResLinearSolver< SparseMatrix<double>, FullVector<double> >;
template class SOFA_GENERAL_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<double>, FullVector<double> >;
template class SOFA_GENERAL_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<Mat<2,2,double> >, FullVector<double> >;
template class SOFA_GENERAL_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<Mat<3,3,double> >, FullVector<double> >;
template class SOFA_GENERAL_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<Mat<4,4,double> >, FullVector<double> >;
template class SOFA_GENERAL_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<Mat<6,6,double> >, FullVector<double> >;
template class SOFA_GENERAL_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<Mat<8,8,double> >, FullVector<double> >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_GENERAL_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<float>, FullVector<float> >;
template class SOFA_GENERAL_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<Mat<2,2,float> >, FullVector<float> >;
template class SOFA_GENERAL_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<Mat<3,3,float> >, FullVector<float> >;
template class SOFA_GENERAL_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<Mat<4,4,float> >, FullVector<float> >;
template class SOFA_GENERAL_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<Mat<6,6,float> >, FullVector<float> >;
template class SOFA_GENERAL_LINEAR_SOLVER_API MinResLinearSolver< CompressedRowSparseMatrix<Mat<8,8,float> >, FullVector<float> >;
#endif

} // namespace linearsolver

} // namespace component

} // namespace sofa

