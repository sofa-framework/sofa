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
#define SOFA_COMPONENT_LINEARSOLVER_MINRESLINEARSOLVER_CPP
#include <sofa/component/linearsolver/iterative/MinResLinearSolver.inl>

#include <sofa/core/ObjectFactory.h>

namespace sofa::component::linearsolver::iterative
{

using namespace sofa::type;
using namespace sofa::defaulttype;
using namespace sofa::linearalgebra;

void registerMinResLinearSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Linear system solver using the MINRES iterative algorithm.")
        .add< MinResLinearSolver< GraphScatteredMatrix, GraphScatteredVector > >(true)
        .add< MinResLinearSolver< FullMatrix<SReal>, FullVector<SReal> > >()
        .add< MinResLinearSolver< SparseMatrix<SReal>, FullVector<SReal> > >()
        .add< MinResLinearSolver< CompressedRowSparseMatrix<SReal>, FullVector<SReal> > >()
        .add< MinResLinearSolver< CompressedRowSparseMatrix<Mat<2, 2, SReal> >, FullVector<SReal> > >()
        .add< MinResLinearSolver< CompressedRowSparseMatrix<Mat<3, 3, SReal> >, FullVector<SReal> > >()
        .add< MinResLinearSolver< CompressedRowSparseMatrix<Mat<4, 4, SReal> >, FullVector<SReal> > >()
        .add< MinResLinearSolver< CompressedRowSparseMatrix<Mat<6, 6, SReal> >, FullVector<SReal> > >()
        .add< MinResLinearSolver< CompressedRowSparseMatrix<Mat<8, 8, SReal> >, FullVector<SReal> > >());
}

template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MinResLinearSolver< GraphScatteredMatrix, GraphScatteredVector >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MinResLinearSolver< FullMatrix<SReal>, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MinResLinearSolver< SparseMatrix<SReal>, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MinResLinearSolver< CompressedRowSparseMatrix<SReal>, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MinResLinearSolver< CompressedRowSparseMatrix<Mat<2,2,SReal> >, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MinResLinearSolver< CompressedRowSparseMatrix<Mat<3,3,SReal> >, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MinResLinearSolver< CompressedRowSparseMatrix<Mat<4,4,SReal> >, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MinResLinearSolver< CompressedRowSparseMatrix<Mat<6,6,SReal> >, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MinResLinearSolver< CompressedRowSparseMatrix<Mat<8,8,SReal> >, FullVector<SReal> >;


} //namespace sofa::component::linearsolver::iterative
