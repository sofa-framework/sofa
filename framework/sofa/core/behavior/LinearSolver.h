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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_BEHAVIOR_LINEARSOLVER_H
#define SOFA_CORE_BEHAVIOR_LINEARSOLVER_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>

namespace sofa
{

namespace core
{

namespace behavior
{

/**
 *  \brief Abstract interface for linear system solvers
 *
 */
class SOFA_CORE_API LinearSolver : public virtual objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(LinearSolver, objectmodel::BaseObject);

    LinearSolver();

    virtual ~LinearSolver();

    /// Reset the current linear system.
    virtual void resetSystem() = 0;

    /// Set the linear system matrix, combining the mechanical M,B,K matrices using the given coefficients
    ///
    /// @todo Should we put this method in a specialized class for mechanical systems, or express it using more general terms (i.e. coefficients of the second order ODE to solve)
    virtual void setSystemMBKMatrix(const MechanicalParams* mparams) = 0;

    /// Set the linear system right-hand term vector, from the values contained in the (Mechanical/Physical)State objects
    virtual void setSystemRHVector(core::MultiVecDerivId v) = 0;

    /// Set the initial estimate of the linear system left-hand term vector, from the values contained in the (Mechanical/Physical)State objects
    /// This vector will be replaced by the solution of the system once solveSystem is called
    virtual void setSystemLHVector(core::MultiVecDerivId v) = 0;

    /// Solve the system as constructed using the previous methods
    virtual void solveSystem() = 0;


    ///
    virtual void init_partial_solve() {serr<<"WARNING : partial_solve is not implemented yet"<<sendl; }

    ///
    virtual void partial_solve(std::list<int>& /*I_last_Disp*/, std::list<int>& /*I_last_Dforce*/, bool /*NewIn*/) {serr<<"WARNING : partial_solve is not implemented yet"<<sendl; }

    /// Invert the system, this method is optional because it's called when solveSystem() is called for the first time
    virtual void invertSystem() {}

    /// Multiply the inverse of the system matrix by the transpose of the given matrix J
    ///
    /// @param result the variable where the result will be added
    /// @param J the matrix J to use
    /// @return false if the solver does not support this operation, of it the system matrix is not invertible
    virtual bool addMInvJt(defaulttype::BaseMatrix* /*result*/, defaulttype::BaseMatrix* /*J*/, double /*fact*/)
    {
        return false;
    }

    /// Multiply the inverse of the system matrix by the transpose of the given matrix, and multiply the result with the given matrix J
    ///
    /// @param result the variable where the result will be added
    /// @param J the matrix J to use
    /// @return false if the solver does not support this operation, of it the system matrix is not invertible
    virtual bool addJMInvJt(defaulttype::BaseMatrix* /*result*/, defaulttype::BaseMatrix* /*J*/, double /*fact*/)
    {
        return false;
    }

    /// Get the linear system matrix, or NULL if this solver does not build it
    virtual defaulttype::BaseMatrix* getSystemBaseMatrix() { return NULL; }

    /// Get the linear system right-hand term vector, or NULL if this solver does not build it
    virtual defaulttype::BaseVector* getSystemRHBaseVector() { return NULL; }

    /// Get the linear system left-hand term vector, or NULL if this solver does not build it
    virtual defaulttype::BaseVector* getSystemLHBaseVector() { return NULL; }

    /// Get the linear system inverse matrix, or NULL if this solver does not build it
    virtual defaulttype::BaseMatrix* getSystemInverseBaseMatrix() { return NULL; }

    /// Read the Matrix solver from a file
    virtual bool readFile(std::istream& /*in*/) { return false;}

    /// Read the Matrix solver from a file
    virtual bool writeFile(std::ostream& /*out*/) {return false;}

    /// Ask the solver to no longer update the system matrix
    virtual void freezeSystemMatrix() { frozen = true; }

    /// Ask the solver to no update the system matrix at the next iteration
    virtual void updateSystemMatrix() { frozen = false; }

    /// Check if this solver handle multiple multiple independent integration groups, placed as child nodes in the scene graph.
    ///
    /// If this is the case, then when collisions occur, the CollisionGroupManager can simply group the interacting groups into new child nodes without creating a new solver to handle them.
    virtual bool isMultiGroup() const
    {
        return false;
    }

protected:

    bool frozen;
};

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
