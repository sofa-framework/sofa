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
#define SOFA_COMPONENT_LINEARSOLVER_MATRIXLINEARSOLVER_CPP
#include <SofaBaseLinearSolver/MatrixLinearSolver.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/MechanicalMatrixVisitor.h>
#include <sofa/simulation/MechanicalVPrintVisitor.h>
#include <sofa/simulation/VelocityThresholdVisitor.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.inl>

#include <stdlib.h>
#include <math.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

using sofa::core::behavior::LinearSolver;
using sofa::core::objectmodel::BaseContext;


template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::resetSystem()
{
    if (!currentGroup->systemMatrix)
    {
        currentGroup->systemMatrix = new GraphScatteredMatrix();
    }
    if (!currentGroup->systemRHVector)
    {
        currentGroup->systemRHVector = new GraphScatteredVector(nullptr,core::VecDerivId::null());
    }
    if (!currentGroup->systemLHVector)
    {
        currentGroup->systemLHVector = new GraphScatteredVector(nullptr,core::VecDerivId::null());
    }
    currentGroup->systemRHVector->reset();
    currentGroup->systemLHVector->reset();
    currentGroup->solutionVecId = core::MultiVecDerivId::null();
    currentGroup->needInvert = true;

}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::resizeSystem(int)
{
    if (!currentGroup->systemMatrix) currentGroup->systemMatrix = new GraphScatteredMatrix();
    if (!currentGroup->systemRHVector) currentGroup->systemRHVector = new GraphScatteredVector(nullptr,core::VecDerivId::null());
    if (!currentGroup->systemLHVector) currentGroup->systemLHVector = new GraphScatteredVector(nullptr,core::VecDerivId::null());
    currentGroup->systemRHVector->reset();
    currentGroup->systemLHVector->reset();
    currentGroup->solutionVecId = core::MultiVecDerivId::null();
    currentGroup->needInvert = true;
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::setSystemMatrix(GraphScatteredMatrix* matrix)
{
    currentGroup->systemMatrix = matrix;
    if (!currentGroup->systemRHVector) currentGroup->systemRHVector = new GraphScatteredVector(nullptr,core::VecDerivId::null());
    if (!currentGroup->systemLHVector) currentGroup->systemLHVector = new GraphScatteredVector(nullptr,core::VecDerivId::null());
    currentGroup->systemRHVector->reset();
    currentGroup->systemLHVector->reset();
    currentGroup->solutionVecId = core::MultiVecDerivId::null();
    currentGroup->needInvert = true;
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::setSystemMBKMatrix(const core::MechanicalParams* mparams)
{
    resetSystem();
    currentGroup->systemMatrix->setMBKFacts(mparams);
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::rebuildSystem(double /*massFactor*/, double /*forceFactor*/)
{
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::setSystemRHVector(core::MultiVecDerivId v)
{
    currentGroup->systemRHVector->set(v);
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::setSystemLHVector(core::MultiVecDerivId v)
{
    currentGroup->solutionVecId = v;
    currentGroup->systemLHVector->set(v);

}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::solveSystem()
{
    if (currentGroup->needInvert)
    {
        this->invert(*currentGroup->systemMatrix);
        currentGroup->needInvert = false;
    }
    this->solve(*currentGroup->systemMatrix, *currentGroup->systemLHVector, *currentGroup->systemRHVector);

}

template<>
GraphScatteredMatrix* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::createMatrix()
{
    return new GraphScatteredMatrix();
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::applyConstraintForce(const sofa::core::ConstraintParams* /*cparams*/, sofa::core::MultiVecDerivId /*dx*/, const defaulttype::BaseVector* /*f*/)
{
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::computeResidual(const core::ExecParams* /*params*/,defaulttype::BaseVector* /*f*/) {
    //todo
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::deleteMatrix(GraphScatteredMatrix* v)
{
    delete v;
}

template<>
GraphScatteredVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::createPersistentVector()
{
    return new GraphScatteredVector(nullptr,core::VecDerivId::null());
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::deletePersistentVector(GraphScatteredVector* v)
{
    delete v;
}

template<>
defaulttype::BaseMatrix* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::getSystemBaseMatrix() { return nullptr; }

template<>
const core::behavior::MultiMatrixAccessor* 
MatrixLinearSolver<GraphScatteredMatrix, GraphScatteredVector, NoThreadManager>::getSystemMultiMatrixAccessor() const
{
    return nullptr;
}

template<>
defaulttype::BaseVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::getSystemRHBaseVector() { return nullptr; }

template<>
defaulttype::BaseVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::getSystemLHBaseVector() { return nullptr; }

// Force template instantiation
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< GraphScatteredMatrix, GraphScatteredVector, NoThreadManager >;
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< FullMatrix<double>, FullVector<double>, NoThreadManager >;
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< FullMatrix<float>, FullVector<float>, NoThreadManager >;
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< SparseMatrix<double>, FullVector<double>, NoThreadManager >;
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< SparseMatrix<float>, FullVector<float>, NoThreadManager >;
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<double>, FullVector<double>, NoThreadManager >;
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<float>, FullVector<float>, NoThreadManager >;
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<2,2,double> >, FullVector<double>, NoThreadManager >;
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<2,2,float> >, FullVector<float>, NoThreadManager >;
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<3,3,double> >, FullVector<double>, NoThreadManager >;
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<3,3,float> >, FullVector<float>, NoThreadManager >;
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<4,4,double> >, FullVector<double>, NoThreadManager >;
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<4,4,float> >, FullVector<float>, NoThreadManager >;
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<6,6,double> >, FullVector<double>, NoThreadManager >;
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<6,6,float> >, FullVector<float>, NoThreadManager >;
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<8,8,double> >, FullVector<double>, NoThreadManager >;
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<8,8,float> >, FullVector<float>, NoThreadManager >;
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< DiagonalMatrix<double>, FullVector<double>, NoThreadManager >;
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< DiagonalMatrix<float>, FullVector<float>, NoThreadManager >;
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< BlockDiagonalMatrix<3,double>, FullVector<double>, NoThreadManager >;
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< RotationMatrix<double>, FullVector<double>, NoThreadManager >;
template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< RotationMatrix<float>, FullVector<float>, NoThreadManager >;

} // namespace linearsolver

} // namespace component

} // namespace sofa
