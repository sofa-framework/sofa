/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include "MatrixLinearSolver.h"
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/MechanicalMatrixVisitor.h>
#include <sofa/simulation/common/MechanicalVPrintVisitor.h>
#include <sofa/simulation/common/VelocityThresholdVisitor.h>
#include <sofa/core/componentmodel/behavior/LinearSolver.h>

#include <stdlib.h>
#include <math.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

using sofa::core::componentmodel::behavior::LinearSolver;
using sofa::core::objectmodel::BaseContext;

void GraphScatteredMatrix::apply(GraphScatteredVector& res, GraphScatteredVector& x)
{
    // matrix-vector product
#if 1
    // new more powerful visitors
    parent->propagateDxAndResetDf(x,res);
    parent->addMBKdx(res,mFact,bFact,kFact); // df = (m M + b B + k K) dx

#else
    parent->propagateDx(x);          // dx = p
    parent->computeDf(res);            // q = K p

    if (kFact != 1.0)
        res *= kFact; // q = k K p

    // apply global Rayleigh damping
    if (mFact == 1.0)
    {
        parent->addMdx(res); // no need to propagate p as dx again
    }
    else if (mFact != 0.0)
    {
        parent->addMdx(res,SolverImpl::VecId(),mFact); // no need to propagate p as dx again
    }
    // q = (m M + k K) p

    /// @TODO: non-rayleigh damping (i.e. the B factor)
#endif

    // filter the product to take the constraints into account
    //
    parent->projectResponse(res);     // q is projected to the constrained space
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::resetSystem()
{
    if (!systemMatrix) systemMatrix = new GraphScatteredMatrix(this);
    if (!systemRHVector) systemRHVector = new GraphScatteredVector(this,VecId());
    if (!systemLHVector) systemLHVector = new GraphScatteredVector(this,VecId());
    systemRHVector->reset();
    systemLHVector->reset();
    solutionVecId = VecId();
    needInvert = true;
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::resizeSystem(int)
{
    resetSystem();
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::setSystemMBKMatrix(double mFact, double bFact, double kFact)
{
    resetSystem();
    systemMatrix->setMBKFacts(mFact, bFact, kFact);
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::setSystemRHVector(VecId v)
{
    systemRHVector->set(v);
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::setSystemLHVector(VecId v)
{
    solutionVecId = v;
    systemLHVector->set(v);
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::solveSystem()
{
    if (needInvert)
    {
        this->invert(*systemMatrix);
        needInvert = false;
    }
    this->solve(*systemMatrix, *systemLHVector, *systemRHVector);
}

template<>
GraphScatteredMatrix* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::createMatrix()
{
    return new GraphScatteredMatrix(this);
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::deleteMatrix(GraphScatteredMatrix* v)
{
    delete v;
}

template<>
GraphScatteredVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::createVector()
{
    return new GraphScatteredVector(this);
}

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::deleteVector(GraphScatteredVector* v)
{
    delete v;
}

template<>
defaulttype::BaseMatrix* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::getSystemBaseMatrix() { return NULL; }

template<>
defaulttype::BaseVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::getSystemRHBaseVector() { return NULL; }

template<>
defaulttype::BaseVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::getSystemLHBaseVector() { return NULL; }

// Force template instantiation
template class MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>;

} // namespace tree

} // namespace simulation

} // namespace sofa
