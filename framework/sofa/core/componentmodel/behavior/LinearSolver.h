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
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_LINEARSOLVER_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_LINEARSOLVER_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/componentmodel/behavior/BaseMechanicalState.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

/**
 *  \brief Abstract interface for linear system solvers
 *
 */
class LinearSolver : public virtual objectmodel::BaseObject
{
public:
    typedef BaseMechanicalState::VecId VecId;

    LinearSolver();

    virtual ~LinearSolver();

    /// Reset the current linear system.
    virtual void resetSystem() = 0;

    /// Set the linear system matrix, combining the mechanical M,B,K matrices using the given coefficients
    ///
    /// @todo Should we put this method in a specialized class for mechanical systems, or express it using more general terms (i.e. coefficients of the second order ODE to solve)
    virtual void setSystemMBKMatrix(double mFact=0.0, double bFact=0.0, double kFact=0.0) = 0;

    /// Set the linear system right-hand term vector, from the values contained in the (Mechanical/Physical)State objects
    virtual void setSystemRHVector(VecId v) = 0;

    /// Set the initial estimate of the linear system left-hand term vector, from the values contained in the (Mechanical/Physical)State objects
    /// This vector will be replaced by the solution of the system once solveSystem is called
    virtual void setSystemLHVector(VecId v) = 0;

    /// Solve the system as constructed using the previous methods
    virtual void solveSystem() = 0;

    /// Get the linear system matrix, or NULL if this solver does not build it
    virtual defaulttype::BaseMatrix* getSystemBaseMatrix() { return NULL; }

    /// Get the linear system right-hand term vector, or NULL if this solver does not build it
    virtual defaulttype::BaseVector* getSystemRHBaseVector() { return NULL; }

    /// Get the linear system left-hand term vector, or NULL if this solver does not build it
    virtual defaulttype::BaseVector* getSystemLHBaseVector() { return NULL; }

    /// Get the linear system inverse matrix, or NULL if this solver does not build it
    virtual defaulttype::BaseMatrix* getSystemInverseBaseMatrix() { return NULL; }

protected:
};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
