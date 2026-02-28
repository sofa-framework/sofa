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

#include <sofa/component/constraint/lagrangian/solver/GenericConstraintSolver.h>
#include <sofa/core/behavior/ConstraintResolution.h>

#include <sofa/simulation/mechanicalvisitor/MechanicalGetConstraintInfoVisitor.h>


namespace sofa::component::constraint::lagrangian::solver
{

/**
 *  \brief This component implements a generic way of preparing system for solvers that doesn't need
 *  a build version of the constraint matrix. Any solver that are based on an unbuilt system should
 *  inherit from this.
 *  This component is purely virtual because doSolve is not defined and needs to be defined in the
 *  inherited class
 */
class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_SOLVER_API UnbuiltConstraintSolver : public GenericConstraintSolver
{
public:
    SOFA_CLASS(UnbuiltConstraintSolver, GenericConstraintSolver);
    
protected:
    UnbuiltConstraintSolver();

public:
    virtual void initializeConstraintProblems() override;

    Data<bool> d_initialGuess; ///< Activate constraint force history to improve convergence (hot start)

protected:
    virtual void doBuildSystem( const core::ConstraintParams *cParams, GenericConstraintProblem * problem, unsigned int numConstraints) override;
    
    virtual void doPreApplyCorrection() override;
    virtual void doPreClearCorrection(const core::ConstraintParams* cparams) override;
    virtual void doPostClearCorrection() override;
    
    ///<
    // Hot-start mechanism types
    typedef core::behavior::BaseLagrangianConstraint::ConstraintBlockInfo ConstraintBlockInfo;
    typedef core::behavior::BaseLagrangianConstraint::PersistentID PersistentID;
    typedef core::behavior::BaseLagrangianConstraint::VecConstraintBlockInfo VecConstraintBlockInfo;
    typedef core::behavior::BaseLagrangianConstraint::VecPersistentID VecPersistentID;

    class ConstraintBlockBuf
    {
    public:
        std::map<PersistentID, int> persistentToConstraintIdMap;
        int nbLines{0}; ///< how many dofs (i.e. lines in the matrix) are used by each constraint
    };

    /// Compute initial guess for constraint forces from previous timestep
    void computeInitialGuess();

    /// Save constraint forces for use as initial guess in next timestep
    void keepContactForcesValue();

    /// Get constraint info (block info and persistent IDs) for hot-start
    void getConstraintInfo(const core::ConstraintParams* cparams);

    // Hot-start data storage
    std::map<core::behavior::BaseLagrangianConstraint*, ConstraintBlockBuf> m_previousConstraints;
    type::vector<SReal> m_previousForces;
    VecConstraintBlockInfo m_constraintBlockInfo;
    VecPersistentID m_constraintIds;
    unsigned int m_numConstraints{0}; ///< Number of constraints from current/previous timestep
};
}
