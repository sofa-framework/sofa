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

#include <sofa/core/objectmodel/BaseComponent.h>
#include <sofa/simulation/task/TaskSchedulerUser.h>
#include <sofa/core/behavior/BaseAnimationLoop.h>
#include <sofa/simulation/CollisionAnimationLoop.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/simulation/VectorOperations.h>

#include <sofa/simulation/fwd.h>


namespace sofa::core
{
class ExecParams;
}


namespace sofa::simulation
{

/**
 *  \brief Default Animation Loop to be created when no AnimationLoop found on simulation::node.
 *
 *
 */
class SOFA_SIMULATION_CORE_API NewtonRaphson : public CollisionAnimationLoop,
                                               public TaskSchedulerUser
{
public:
    typedef sofa::core::objectmodel::BaseContext BaseContext;
    SOFA_CLASS2(NewtonRaphson, CollisionAnimationLoop, TaskSchedulerUser);

    virtual void init() override;
    virtual void integrate(const core::ExecParams* params, SReal dt) override;

    Data<unsigned int> d_maxNbIterationsNewton;
    Data<unsigned int> d_maxNbIterationsLineSearch;
    Data<SReal> d_newtonStepSize;
    Data<SReal> d_lineSearchReductionRate;
    Data<SReal> d_lineSearchArmijoFactor;
    Data<SReal> d_residueThreshold;
    Data<SReal> d_currentResidue;
    Data<bool> d_alwaysAdvanceNewton;
    Data<bool> d_performCollisionDetectionEachNewtonStep;
    Data<bool> d_solveVelocityConstraintFirst; ///< solve separately velocity constraint violations before position constraint violations
    Data<bool> d_parallelODESolving; ///< If true, solves all the ODEs in parallel during the free motion step.

protected:
    NewtonRaphson();
    ~NewtonRaphson() override;

    SingleLink<NewtonRaphson, sofa::core::behavior::ConstraintSolver, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> l_constraintSolver;

    void preSolve(const core::ExecParams* paramsms, core::ConstraintParams* cparams, common::VectorOperations* vop, common::MechanicalOperations* mop, SReal dt, Node* node);
    void newtonSolve(const core::ExecParams* params, core::ConstraintParams* cparams, common::VectorOperations* vop, common::MechanicalOperations* mop,  SReal dt, sofa::core::MultiVecCoordId xId, sofa::core::MultiVecDerivId vId, Node* node);
    void postSolve(const core::ExecParams* params, core::ConstraintParams* cparams, common::VectorOperations* vop, common::MechanicalOperations* mop, SReal startTime, SReal dt, Node* node);

    void computeCollision(const core::ExecParams* params) override;
    void computeGeometricalStiffness(const core::MechanicalParams* mparams, const core::ConstraintParams* cparams, Node* node);
    void solveConstraints(const core::MechanicalParams* mparams, const core::ConstraintParams* cparams, common::VectorOperations* vop, common::MechanicalOperations* mop, sofa::core::MultiVecCoordId xId, sofa::core::MultiVecDerivId vId, SReal dt);

    bool m_solveConstraints;
    core::ConstraintParams m_cparams;


};

} // namespace sofa::simulation
