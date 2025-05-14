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

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/behavior/StateAccessor.h>

#include <sofa/defaulttype/fwd.h>
#include <sofa/core/behavior/fwd.h>
#include <sofa/core/fwd.h>

namespace sofa::core::behavior
{

/**
 * Interface to apply a zero Dirichlet boundary condition on a matrix
 *
 * If K is a matrix to apply a zero Dirichlet boundary condition:
 *  K_ii = 1
 *  K_ij = 0 for i != j
 *  K_ji = 0 for i != j
 */
struct ZeroDirichletCondition
{
    virtual ~ZeroDirichletCondition() = default;
    /**
     * Zero out a row and a column of a matrix. The element at the
     * intersection of the row and the column is set to 1.
     */
    virtual void discardRowCol(sofa::Index /*row*/, sofa::Index /*col*/) {}
};

/**
*  \brief Component computing projective constraints within a simulated body.
*
*  This class define the abstract API common to all projective constraints.
*  A BaseConstraint computes constraints applied to one or more simulated body
*  given its current position and velocity.
*
*  Constraints can be internal to a given body (attached to one MechanicalState,
*  see the Constraint class), or link several bodies together (such as contacts,
*  see the InteractionConstraint class).
*
*/
class SOFA_CORE_API BaseProjectiveConstraintSet : public virtual StateAccessor
{
public:
    SOFA_ABSTRACT_CLASS(BaseProjectiveConstraintSet, objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(BaseProjectiveConstraintSet)
protected:
    BaseProjectiveConstraintSet()
        : group(initData(&group, 0, "group", "ID of the group containing this constraint. This ID is used to specify which constraints are solved by which solver, by specifying in each solver which groups of constraints it should handle."))
    {
    }

    ~BaseProjectiveConstraintSet() override {}
	
    virtual type::vector< core::BaseState* > doGetModels() = 0;
    virtual void doProjectResponse(const MechanicalParams* mparams, MultiVecDerivId dxId) = 0;
    virtual void doProjectResponse(const MechanicalParams* /*mparams*/, double **) { };
    virtual void doProjectJacobianMatrix(const MechanicalParams* mparams, MultiMatrixDerivId cId) = 0;
    virtual void doProjectVelocity(const MechanicalParams* mparams, MultiVecDerivId vId) = 0;
    virtual void doProjectPosition(const MechanicalParams* mparams, MultiVecCoordId xId) = 0;

    virtual void doApplyConstraint(const MechanicalParams* /*mparams*/, const behavior::MultiMatrixAccessor* /*matrix*/) {}
    virtual void doApplyConstraint(const MechanicalParams* /*mparams*/, linearalgebra::BaseVector* /*vector*/, const behavior::MultiMatrixAccessor* /*matrix*/) {}
    virtual void doApplyConstraint(sofa::core::behavior::ZeroDirichletCondition* /*matrix*/);

public:
    /// Get the ID of the group containing this constraint.
    /// This ID is used to specify which constraints are solved by which solver, by specifying in each solver which groups of constraints it should handle.
    int getGroup() const { return group.getValue(); }

    /// Set the ID of the group containing this constraint.
    /// This ID is used to specify which constraints are solved by which solver, by specifying in each solver which groups of constraints it should handle.
    void setGroup(int g) { group.setValue(g); }


    /// Return the lists of models this constraint applies to. 
    virtual type::vector< core::BaseState* > getModels() final {
      return this->doGetModels();
    }

    /// @name Vector operations
    /// @{

    /// Project dx to constrained space (dx models an acceleration).
    /// \param dxId output vector
    virtual void projectResponse(const MechanicalParams* mparams, MultiVecDerivId dxId) final {
      this->doProjectResponse(mparams, dxId);
    }

    /// Project the L matrix of the Lagrange Multiplier equation system.
    /// \param cId output vector
    virtual void projectJacobianMatrix(const MechanicalParams* mparams, MultiMatrixDerivId cId) final {
      this->doProjectJacobianMatrix(mparams, cId);
    }

    /// Project v to constrained space (v models a velocity).
    /// \param vId output vector
    virtual void projectVelocity(const MechanicalParams* mparams, MultiVecDerivId vId) final {
      this->doProjectVelocity(mparams, vId);
    }

    /// Project x to constrained space (x models a position).
    /// \param xId output vector
    virtual void projectPosition(const MechanicalParams* mparams, MultiVecCoordId xId) final {
      this->doProjectPosition(mparams, xId);
    }

    /// @}


    /// @name Matrix operations
    /// @{

    /// Project the compliance Matrix to constrained space.
    virtual void projectResponse(const MechanicalParams* mparams, double ** W) final {
      this->doProjectResponse(mparams, W);
    }

    /// Project the global Mechanical Matrix to constrained space using offset parameter
    virtual void applyConstraint(const MechanicalParams* mparams, const behavior::MultiMatrixAccessor* matrix) final {
      this->doApplyConstraint(mparams, matrix);
    }

    /// Project the global Mechanical Vector to constrained space using offset parameter
    virtual void applyConstraint(const MechanicalParams* mparams, linearalgebra::BaseVector* vector, const behavior::MultiMatrixAccessor* matrix) final {
      this->doApplyConstraint(mparams, vector, matrix);
    }

    /** Project the given matrix (Experimental API).
      Replace M with PMP, where P is the projection matrix corresponding to the projectResponse method. Contrary to applyConstraint(), the diagonal blocks of the result are not reset to the identity.
      Typically, M is the (generalized) mass matrix of the whole system, offset is the starting index of the local state in this global matrix, and P is the identity matrix with a block on the diagonal replaced by the projection matrix.
      If M is the matrix of the local state, then offset should be 0.
      */
    virtual void projectMatrix( sofa::linearalgebra::BaseMatrix* /*M*/, unsigned /*offset*/ ) 
    { 
        msg_error() << "projectMatrix not implemented, projection will not be handled appropriately";
    }

    /// Project the global matrix to constrained space by using the ZeroDirichletCondition interface
    /// It allows to define what rows and columns to discard for the projection.
    virtual void applyConstraint(sofa::core::behavior::ZeroDirichletCondition* matrix) final {
      this->doApplyConstraint(matrix);
    }

    /// @}

protected:
    Data<int> group; ///< ID of the group containing this constraint. This ID is used to specify which constraints are solved by which solver, by specifying in each solver which groups of constraints it should handle.

public:

    bool insertInNode( objectmodel::BaseNode* node ) override;
    bool removeInNode( objectmodel::BaseNode* node ) override;

};

} // namespace sofa::core::behavior
