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
 *  \brief Abstract base class (as type identifier) for linear system solvers without any API
 *
 */
class SOFA_CORE_API BaseLinearSolver : public objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(BaseLinearSolver, objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(BaseLinearSolver)

    /// Check if this solver handle multiple independent integration groups, placed as child nodes in the scene graph.
    /// If this is the case, then when collisions occur, the CollisionGroupManager can simply group the interacting groups into new child nodes without creating a new solver to handle them.
    virtual bool isMultiGroup() const
    {
        return false;
    }

    virtual bool insertInNode( objectmodel::BaseNode* node );
    virtual bool removeInNode( objectmodel::BaseNode* node );

}; // class BaseLinearSolver




/**
 *  \brief Abstract interface for linear system solvers
 *
 */
class SOFA_CORE_API LinearSolver : public BaseLinearSolver
{
public:
    SOFA_ABSTRACT_CLASS(LinearSolver, BaseLinearSolver);
    SOFA_BASE_CAST_IMPLEMENTATION(LinearSolver)
protected:
    LinearSolver();

    virtual ~LinearSolver();
public:
    /// Reset the current linear system.
    virtual void resetSystem() = 0;

    /// Set the linear system matrix, combining the mechanical M,B,K matrices using the given coefficients
    ///
    /// @todo Should we put this method in a specialized class for mechanical systems, or express it using more general terms (i.e. coefficients of the second order ODE to solve)
    virtual void setSystemMBKMatrix(const MechanicalParams* mparams) = 0;

    /// Rebuild the system using a mass and force factor
    /// Experimental API used to investigate convergence issues.
    virtual void rebuildSystem(SReal /*massFactor*/, SReal /*forceFactor*/){}

    /// Indicate if the solver update the system in parallel
    virtual bool isAsyncSolver() { return false; }

    /// Indicate if the solver updated the system after the last call of setSystemMBKMatrix (should return true if isParallelSolver return false)
    virtual bool hasUpdatedMatrix() { return true; }

    /// This function is use for the preconditioner it must be called at each time step event if setSystemMBKMatrix is not called
    virtual void updateSystemMatrix() {}

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
    /// @param fact integrator parameter
    /// @return false if the solver does not support this operation, of it the system matrix is not invertible
    virtual bool addMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, SReal fact)
    {
        SOFA_UNUSED(result);
        SOFA_UNUSED(J);
        SOFA_UNUSED(fact);
        return false;
    }

    /// Build the jacobian of the constraints using a visitor
    ///
    /// @param result the variable where the result will be added
    /// @param fact integrator parameter
    /// @return false if the solver does not support this operation, of it the system matrix is not invertible
    virtual bool buildComplianceMatrix(defaulttype::BaseMatrix* result, SReal fact)
    {
        SOFA_UNUSED(result);
        SOFA_UNUSED(fact);
        serr << "Error buildComplianceMatrix has not been implemented" << sendl;
        return false;
    }

    /// Apply the contactforce dx = Minv * J * f and store the resut in VecId
    virtual void applyContactForce(const defaulttype::BaseVector* /*f*/,SReal /*positionFactor*/,SReal /*velocityFactor*/) {
        serr << "Error applyContactForce has not been implemented" << sendl;
    }

    /// Compute the residual in the newton iterations due to the constraints forces
    /// i.e. compute mparams->dF() = J^t lambda
    /// the result is written in mparams->dF()
    virtual void computeResidual(const core::ExecParams* /*params*/, defaulttype::BaseVector* /*f*/) {
        serr << "Error applyContactForce has not been implemented" << sendl;
    }


    /// Multiply the inverse of the system matrix by the transpose of the given matrix, and multiply the result with the given matrix J
    ///
    /// @param result the variable where the result will be added
    /// @param J the matrix J to use
    /// @param fact integrator parameter
    /// @return false if the solver does not support this operation, of it the system matrix is not invertible
    virtual bool addJMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, SReal fact)
    {
        SOFA_UNUSED(result);
        SOFA_UNUSED(J);
        SOFA_UNUSED(fact);
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



protected:

    bool frozen;
};

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
