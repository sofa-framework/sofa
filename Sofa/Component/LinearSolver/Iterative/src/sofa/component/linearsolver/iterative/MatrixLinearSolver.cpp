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
#define SOFA_COMPONENT_LINEARSOLVER_MATRIXLINEARSOLVER_CPP
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.inl>
#include <sofa/component/linearsolver/iterative/MatrixFreeSystem[GraphScattered].h>

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/linearalgebra/BlockDiagonalMatrix.h>


namespace sofa::component::linearsolver
{

using sofa::core::behavior::LinearSolver;
using sofa::core::objectmodel::BaseContext;

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::applyConstraintForce(const sofa::core::ConstraintParams* /*cparams*/, sofa::core::MultiVecDerivId /*dx*/, const linearalgebra::BaseVector* /*f*/)
{
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::computeResidual(const core::ExecParams* /*params*/,linearalgebra::BaseVector* /*f*/) {
    //todo
}

template<>
GraphScatteredVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::createPersistentVector()
{
    return new GraphScatteredVector(nullptr,core::VecDerivId::null());
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::checkLinearSystem()
{
    doCheckLinearSystem<linearsystem::MatrixFreeSystem<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector> >();
}

template<>
bool MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::addJMInvJtLocal(
    GraphScatteredMatrix* M,
    MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::ResMatrixType* result, const
    MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::JMatrixType* J, const SReal fact)
{
    return singleThreadAddJMInvJtLocal(M, result, J, fact);
}

// Force template instantiation
using namespace sofa::linearalgebra;

template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< GraphScatteredMatrix, GraphScatteredVector, NoThreadManager >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< FullMatrix<SReal>, FullVector<SReal>, NoThreadManager >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< SparseMatrix<SReal>, FullVector<SReal>, NoThreadManager >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< CompressedRowSparseMatrix<SReal>, FullVector<SReal>, NoThreadManager >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<2,2,SReal> >, FullVector<SReal>, NoThreadManager >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<3,3,SReal> >, FullVector<SReal>, NoThreadManager >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<4,4,SReal> >, FullVector<SReal>, NoThreadManager >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<6,6,SReal> >, FullVector<SReal>, NoThreadManager >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<8,8,SReal> >, FullVector<SReal>, NoThreadManager >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< DiagonalMatrix<SReal>, FullVector<SReal>, NoThreadManager >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< BlockDiagonalMatrix<3,SReal>, FullVector<SReal>, NoThreadManager >;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< RotationMatrix<SReal>, FullVector<SReal>, NoThreadManager >;

} // namespace sofa::component::linearsolver
