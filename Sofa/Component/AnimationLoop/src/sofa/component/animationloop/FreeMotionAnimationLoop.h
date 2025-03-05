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
#include <sofa/component/animationloop/config.h>

#include <sofa/simulation/CollisionAnimationLoop.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/objectmodel/lifecycle/RenamedData.h>

namespace sofa::core::behavior
{
    class ConstraintSolver;
}

namespace sofa::component::animationloop
{

class SOFA_COMPONENT_ANIMATIONLOOP_API FreeMotionAnimationLoop : public sofa::simulation::CollisionAnimationLoop
{
public:
    SOFA_CLASS(FreeMotionAnimationLoop, sofa::simulation::CollisionAnimationLoop);

public:
    void step (const sofa::core::ExecParams* params, SReal dt) override;
    void init() override;


    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ANIMATIONLOOP()
    sofa::core::objectmodel::lifecycle::RenamedData<bool> m_solveVelocityConstraintFirst;

    Data<bool> d_solveVelocityConstraintFirst; ///< solve separately velocity constraint violations before position constraint violations
    Data<bool> d_threadSafeVisitor; ///< If true, do not use realloc and free visitors in fwdInteractionForceField.
    Data<bool> d_parallelCollisionDetectionAndFreeMotion; ///< If true, executes free motion step and collision detection step in parallel.
    Data<bool> d_parallelODESolving; ///< If true, solves all the ODEs in parallel during the free motion step.

protected:
    FreeMotionAnimationLoop();
    ~FreeMotionAnimationLoop() override ;

    ///< The ConstraintSolver used in this animation loop (required)
    SingleLink<FreeMotionAnimationLoop, sofa::core::behavior::ConstraintSolver, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> l_constraintSolver;

    void computeFreeMotionAndCollisionDetection(const sofa::core::ExecParams* params, const core::ConstraintParams& cparams, SReal dt,
                                         sofa::core::MultiVecId pos,
                                         sofa::core::MultiVecId freePos,
                                         sofa::core::MultiVecDerivId freeVel,
                                         simulation::common::MechanicalOperations* mop);

    void computeFreeMotion(const sofa::core::ExecParams* params, const core::ConstraintParams& cparams, SReal dt,
                                         sofa::core::MultiVecId pos,
                                         sofa::core::MultiVecId freePos,
                                         sofa::core::MultiVecDerivId freeVel,
                                         simulation::common::MechanicalOperations* mop);
};

} // namespace sofa::component::animationloop
