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
#include <sofa/simulation/AnimateVisitor.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/CollisionVisitor.h>

#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/CollisionBeginEvent.h>
#include <sofa/simulation/CollisionEndEvent.h>
#include <sofa/simulation/IntegrateBeginEvent.h>
#include <sofa/simulation/IntegrateEndEvent.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/BehaviorModel.h>
#include <sofa/core/behavior/BaseInteractionForceField.h>
#include <sofa/core/collision/Pipeline.h>

#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

#include <sofa/simulation/mechanicalvisitor/MechanicalResetConstraintVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalResetConstraintVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalBeginIntegrationVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalBeginIntegrationVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalProjectPositionAndVelocityVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalProjectPositionAndVelocityVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyPositionAndVelocityVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalPropagateOnlyPositionAndVelocityVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalEndIntegrationVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalEndIntegrationVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalAccumulateMatrixDeriv.h>
using sofa::simulation::mechanicalvisitor::MechanicalAccumulateMatrixDeriv;

#include <sofa/simulation/mechanicalvisitor/MechanicalBuildConstraintMatrix.h>
using sofa::simulation::mechanicalvisitor::MechanicalBuildConstraintMatrix;

using namespace sofa::core;

namespace sofa::simulation
{


AnimateVisitor::AnimateVisitor(const core::ExecParams* params, SReal dt)
    : Visitor(params)
    , dt(dt)
    , firstNodeVisited(false)
{
}

void AnimateVisitor::fwdInteractionForceField(simulation::Node*, core::behavior::BaseInteractionForceField* obj)
{
    helper::ScopedAdvancedTimer timer("InteractionFF", obj);

    const MultiVecDerivId   ffId      = VecDerivId::externalForce();
    MechanicalParams mparams;
    mparams.setDt(this->dt);
    obj->addForce(&mparams, ffId);
}

void AnimateVisitor::processCollisionPipeline(simulation::Node* node, core::collision::Pipeline* obj)
{
    helper::ScopedAdvancedTimer collisionTimer("Collision", obj);

    {
        helper::ScopedAdvancedTimer timer("begin collision", obj);
        CollisionBeginEvent evBegin;
        PropagateEventVisitor eventPropagation( params, &evBegin);
        eventPropagation.execute(node->getContext());
    }

    CollisionVisitor act(this->params);
    node->execute(&act);

    {
        helper::ScopedAdvancedTimer timer("end collision", obj);
        CollisionEndEvent evEnd;
        PropagateEventVisitor eventPropagation( params, &evEnd);
        eventPropagation.execute(node->getContext());
    }
}

Visitor::Result AnimateVisitor::processNodeTopDown(simulation::Node* node)
{
    if (!node->isActive()) return Visitor::RESULT_PRUNE;
    if (node->isSleeping()) return Visitor::RESULT_PRUNE;
    if (!firstNodeVisited)
    {
        firstNodeVisited=true;
        
        sofa::core::ConstraintParams cparams(*this->params);
        MechanicalResetConstraintVisitor resetConstraint(&cparams);
        node->execute(&resetConstraint);
    }

    if (dt == 0) setDt(node->getDt());
    else node->setDt(dt);

    if (node->collisionPipeline != nullptr)
    {
        processCollisionPipeline(node, node->collisionPipeline);
    }
    if (!node->solver.empty() )
    {
        sofa::helper::AdvancedTimer::StepVar timer("Mechanical",node);
        const SReal nextTime = node->getTime() + dt;
        {
            IntegrateBeginEvent evBegin;
            PropagateEventVisitor eventPropagation( this->params, &evBegin);
            eventPropagation.execute(node);
        }

        MechanicalBeginIntegrationVisitor beginVisitor(this->params, dt);
        node->execute(&beginVisitor);

        sofa::core::MechanicalParams m_mparams(*this->params);
        m_mparams.setDt(dt);

        core::ConstraintParams cparams;
        {
            unsigned int constraintId=0;
            MechanicalBuildConstraintMatrix buildConstraintMatrix(&cparams, core::MatrixDerivId::constraintJacobian(), constraintId );
            buildConstraintMatrix.execute(node);
        }

        {
            MechanicalAccumulateMatrixDeriv accumulateMatrixDeriv(&cparams, core::MatrixDerivId::constraintJacobian());
            accumulateMatrixDeriv.execute(node);
        }

        for( unsigned i=0; i<node->solver.size(); i++ )
        {
            ctime_t t0 = begin(node, node->solver[i]);
            node->solver[i]->solve(params, getDt());
            end(node, node->solver[i], t0);
        }

        MechanicalProjectPositionAndVelocityVisitor(&m_mparams, nextTime,
                                                    sofa::core::VecCoordId::position(), sofa::core::VecDerivId::velocity()
                                                    ).execute( node );
        MechanicalPropagateOnlyPositionAndVelocityVisitor(&m_mparams, nextTime,
                                                          VecCoordId::position(),
                                                          VecDerivId::velocity()).execute( node );

        MechanicalEndIntegrationVisitor endVisitor(this->params, dt);
        node->execute(&endVisitor);

        {
            IntegrateEndEvent evBegin;
            PropagateEventVisitor eventPropagation(this->params, &evBegin);
            eventPropagation.execute(node);
        }

        return RESULT_PRUNE;
    }

    // process InteractionForceFields
    for_each(this, node, node->interactionForceField, &AnimateVisitor::fwdInteractionForceField);
    return RESULT_CONTINUE;
}

} // namespace sofa::simulation

