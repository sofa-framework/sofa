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
#pragma once
#include <sofa/core/config.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/core/MultiVecId.h>

namespace sofa::core::behavior
{

/**
 * Base class for components storing and assembling a linear system represented as a matrix.
 * The matrix data structure is defined in derived classes.
 */
class SOFA_CORE_API BaseMatrixLinearSystem : public virtual core::objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(BaseMatrixLinearSystem, core::objectmodel::BaseObject);

protected:
    BaseMatrixLinearSystem();

public:

    /// Size of the linear system
    Data< sofa::type::Vec2u > d_matrixSize;

    Data< bool > d_enableAssembly;

    /// Returns the system matrix as a sofa::linearalgebra::BaseMatrix*
    virtual linearalgebra::BaseMatrix* getSystemBaseMatrix() const { return nullptr; }

    virtual linearalgebra::BaseVector* getSystemRHSBaseVector() const { return nullptr; }
    virtual linearalgebra::BaseVector* getSystemSolutionBaseVector() const { return nullptr; }

    /// Construct and assemble the linear system matrix
    virtual void buildSystemMatrix(const core::MechanicalParams* mparams);

    sofa::type::Vec2u getMatrixSize() const { return d_matrixSize.getValue(); }

    /// Set the size of the matrix to n x n, and the size of RHS and solution to n
    virtual void resizeSystem(sofa::Size n) = 0;

    virtual void clearSystem() = 0;

    /// Assemble the right-hand side of the linear system from the values contained in the (Mechanical/Physical)State objects
    virtual void setRHS(core::MultiVecDerivId v) = 0;

    /// Set the initial estimate of the linear system solution vector, from the values contained in the (Mechanical/Physical)State objects
    /// This vector will be replaced by the solution of the system once the system is solved
    virtual void setSystemSolution(core::MultiVecDerivId v) = 0;

    virtual void dispatchSystemSolution(core::MultiVecDerivId v) = 0;
    virtual void dispatchSystemRHS(core::MultiVecDerivId v) = 0;

protected:
    virtual void preAssembleSystem(const core::MechanicalParams* /*mparams*/);
    virtual void assembleSystem(const core::MechanicalParams* /*mparams*/);
    virtual void postAssembleSystem(const core::MechanicalParams* /*mparams*/) {}
};

/// This tag is used to differentiate the matrix components that have been instantiated
/// automatically by a BaseMatrixLinearSystem, from the user design.
static constexpr const char* tagSetupByMatrixLinearSystem =  "SetupByMatrixLinearSystem";

} //namespace sofa::core::behavior
