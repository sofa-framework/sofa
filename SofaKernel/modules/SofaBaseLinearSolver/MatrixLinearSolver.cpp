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
//    serr << "resetSystem()" << sendl;
    for (unsigned int g=0, nbg = getNbGroups(); g < nbg; ++g)
    {
        setGroup(g);
        if (!currentGroup->systemMatrix)
        {
//            serr << "new systemMatrix" << sendl;
            currentGroup->systemMatrix = new GraphScatteredMatrix();
        }
        if (!currentGroup->systemRHVector)
        {
//            serr << "new systemRHVector" << sendl;
            currentGroup->systemRHVector = new GraphScatteredVector(NULL,core::VecDerivId::null());
        }
        if (!currentGroup->systemLHVector)
        {
//            serr << "new systemLHVector" << sendl;
            currentGroup->systemLHVector = new GraphScatteredVector(NULL,core::VecDerivId::null());
        }
        currentGroup->systemRHVector->reset();
        currentGroup->systemLHVector->reset();
        currentGroup->solutionVecId = core::MultiVecDerivId::null();
        currentGroup->needInvert = true;
    }
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::resizeSystem(int)
{
//    serr << "resizeSystem()" << sendl;
    if (!currentGroup->systemMatrix) currentGroup->systemMatrix = new GraphScatteredMatrix();
    if (!currentGroup->systemRHVector) currentGroup->systemRHVector = new GraphScatteredVector(NULL,core::VecDerivId::null());
    if (!currentGroup->systemLHVector) currentGroup->systemLHVector = new GraphScatteredVector(NULL,core::VecDerivId::null());
    currentGroup->systemRHVector->reset();
    currentGroup->systemLHVector->reset();
    currentGroup->solutionVecId = core::MultiVecDerivId::null();
    currentGroup->needInvert = true;
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::setSystemMatrix(GraphScatteredMatrix* matrix)
{
//    serr << "resizeSystem()" << sendl;
    currentGroup->systemMatrix = matrix;
    if (!currentGroup->systemRHVector) currentGroup->systemRHVector = new GraphScatteredVector(NULL,core::VecDerivId::null());
    if (!currentGroup->systemLHVector) currentGroup->systemLHVector = new GraphScatteredVector(NULL,core::VecDerivId::null());
    currentGroup->systemRHVector->reset();
    currentGroup->systemLHVector->reset();
    currentGroup->solutionVecId = core::MultiVecDerivId::null();
    currentGroup->needInvert = true;
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::setSystemMBKMatrix(const core::MechanicalParams* mparams)
{
//    serr << "setSystemMBKMatrix(" << mparams->mFactor() << ", " << mparams->bFactor() << ", " << mparams->kFactor() << ")" << sendl;
    createGroups(mparams);
    resetSystem();
    for (unsigned int g=0, nbg = getNbGroups(); g < nbg; ++g)
    {
        setGroup(g);
        currentGroup->systemMatrix->setMBKFacts(mparams);
    }
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::rebuildSystem(double /*massFactor*/, double /*forceFactor*/)
{
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::setSystemRHVector(core::MultiVecDerivId v)
{
//    serr << "setSystemRHVector(" << v << ")" << sendl;
    for (unsigned int g=0, nbg = getNbGroups(); g < nbg; ++g)
    {
        setGroup(g);
        currentGroup->systemRHVector->set(v);
    }
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::setSystemLHVector(core::MultiVecDerivId v)
{
//    serr << "setSystemLHVector(" << v << ")" << sendl;
    for (unsigned int g=0, nbg = getNbGroups(); g < nbg; ++g)
    {
        setGroup(g);
        currentGroup->solutionVecId = v;
        currentGroup->systemLHVector->set(v);
    }
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::solveSystem()
{
//    serr << "solveSystem()" << sendl;
    for (unsigned int g=0, nbg = getNbGroups(); g < nbg; ++g)
    {
        setGroup(g);
        if (currentGroup->needInvert)
        {
//            serr << "->invert(M)" << sendl;
            this->invert(*currentGroup->systemMatrix);
//            serr << "<<invert(M)" << sendl;
            currentGroup->needInvert = false;
        }
//        serr << "->solve(M, " << currentGroup->systemLHVector->id()  << ", " << currentGroup->systemRHVector->id() << ")" << sendl;
        this->solve(*currentGroup->systemMatrix, *currentGroup->systemLHVector, *currentGroup->systemRHVector);
//        serr << "<<solve(M, " << currentGroup->systemLHVector->id()  << ", " << currentGroup->systemRHVector->id() << ")" << sendl;
    }
}

template<>
GraphScatteredMatrix* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::createMatrix()
{
    return new GraphScatteredMatrix();
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::applyContactForce(const defaulttype::BaseVector* /*f*/,double /*positionFactor*/,double /*velocityFactor*/) {
//    FullVector<Real> temporaryVector;
//    temporaryVector.resize(currentGroup->systemMatrix->rowSize());
//    internalData.projectForceInConstraintSpace(&temporaryVector,f);

//    std::cout << "temporaryVector1 = " << temporaryVector << std::endl;

//    executeVisitor( simulation::MechanicalMultiVectorFromBaseVectorVisitor(core::ExecParams::defaultInstance(), currentGroup->systemRHVector->id(), &temporaryVector, &(currentGroup->matrixAccessor)) );

//    this->solve(*currentGroup->systemMatrix,*currentGroup->systemLHVector,*currentGroup->systemRHVector);

//    executeVisitor( simulation::MechanicalMultiVectorToBaseVectorVisitor(core::ExecParams::defaultInstance(), currentGroup->systemLHVector->id(), &temporaryVector, &(currentGroup->matrixAccessor)) );

//    std::cout << "temporaryVector2 = " << temporaryVector << std::endl;

//    executeVisitor(simulation::MechanicalIntegrateConstraintsVisitor(core::ExecParams::defaultInstance(),&temporaryVector,positionFactor,velocityFactor,&(currentGroup->matrixAccessor)));
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
    return new GraphScatteredVector(NULL,core::VecDerivId::null());
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::deletePersistentVector(GraphScatteredVector* v)
{
    delete v;
}

template<>
defaulttype::BaseMatrix* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::getSystemBaseMatrix() { return NULL; }

template<>
defaulttype::BaseVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::getSystemRHBaseVector() { return NULL; }

template<>
defaulttype::BaseVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::getSystemLHBaseVector() { return NULL; }

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
