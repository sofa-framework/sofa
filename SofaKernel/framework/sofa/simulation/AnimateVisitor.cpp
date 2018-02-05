/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <sofa/simulation/AnimateVisitor.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/CollisionVisitor.h>

#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/CollisionBeginEvent.h>
#include <sofa/simulation/CollisionEndEvent.h>
#include <sofa/simulation/IntegrateBeginEvent.h>
#include <sofa/simulation/IntegrateEndEvent.h>
#include <sofa/simulation/PropagateEventVisitor.h>


#include <sofa/helper/AdvancedTimer.h>

//#include "MechanicalIntegration.h"

using namespace sofa::core;

namespace sofa
{

namespace simulation
{


AnimateVisitor::AnimateVisitor(const core::ExecParams* params, SReal dt)
    : Visitor(params)
    , dt(dt)
    , firstNodeVisited(false)
{
}

AnimateVisitor::AnimateVisitor(const core::ExecParams* params)
    : Visitor(params)
    , dt(0)
    , firstNodeVisited(false)
{
}

void AnimateVisitor::processBehaviorModel(simulation::Node*, core::BehaviorModel* obj)
{
    sofa::helper::AdvancedTimer::stepBegin("BehaviorModel",obj);

    obj->updatePosition(getDt());
    sofa::helper::AdvancedTimer::stepEnd("BehaviorModel",obj);
}

void AnimateVisitor::fwdInteractionForceField(simulation::Node*, core::behavior::BaseInteractionForceField* obj)
{
    //cerr<<"AnimateVisitor::IFF "<<obj->getName()<<endl;
    sofa::helper::AdvancedTimer::stepBegin("InteractionFF",obj);

    MultiVecDerivId   ffId      = VecDerivId::externalForce();
    MechanicalParams mparams; // = MechanicalParams::defaultInstance();
    mparams.setDt(this->dt);
    obj->addForce(&mparams, ffId);

    sofa::helper::AdvancedTimer::stepEnd("InteractionFF",obj);
}

void AnimateVisitor::processCollisionPipeline(simulation::Node* node, core::collision::Pipeline* obj)
{
     sofa::helper::AdvancedTimer::stepBegin("Collision",obj);

    sofa::helper::AdvancedTimer::stepBegin("begin collision",obj);
    {
        CollisionBeginEvent evBegin;
        PropagateEventVisitor eventPropagation( params, &evBegin);
        eventPropagation.execute(node->getContext());
    }
    sofa::helper::AdvancedTimer::stepEnd("begin collision",obj);

    CollisionVisitor act(this->params);
    node->execute(&act);

    sofa::helper::AdvancedTimer::stepBegin("end collision",obj);
    {
        CollisionEndEvent evEnd;
        PropagateEventVisitor eventPropagation( params, &evEnd);
        eventPropagation.execute(node->getContext());
    }
    sofa::helper::AdvancedTimer::stepEnd("end collision",obj);

    sofa::helper::AdvancedTimer::stepEnd("Collision",obj);
}

void AnimateVisitor::processOdeSolver(simulation::Node* node, core::behavior::OdeSolver* solver)
{
    sofa::helper::AdvancedTimer::stepBegin("Mechanical",node);
    /*    MechanicalIntegrationVisitor act(getDt());
        node->execute(&act);*/
//  cerr<<"AnimateVisitor::processOdeSolver "<<solver->getName()<<endl;
    solver->solve(params, getDt());
    sofa::helper::AdvancedTimer::stepEnd("Mechanical",node);
}

Visitor::Result AnimateVisitor::processNodeTopDown(simulation::Node* node)
{
    if (!node->isActive()) return Visitor::RESULT_PRUNE;
    if (node->isSleeping()) return Visitor::RESULT_PRUNE;
    if (!firstNodeVisited)
    {
        firstNodeVisited=true;
        sofa::core::MechanicalParams mparams(*this->params);
        mparams.setDt(dt);
        MechanicalResetConstraintVisitor resetConstraint(&mparams);
        node->execute(&resetConstraint);
    }

    if (dt == 0) setDt(node->getDt());
    else node->setDt(dt);

    if (node->collisionPipeline != NULL)
    {
        processCollisionPipeline(node, node->collisionPipeline);
    }
    if (!node->solver.empty() )
    {
        sofa::helper::AdvancedTimer::StepVar timer("Mechanical",node);
        SReal nextTime = node->getTime() + dt;
        {
            IntegrateBeginEvent evBegin;
            PropagateEventVisitor eventPropagation( this->params, &evBegin);
            eventPropagation.execute(node);
        }

        MechanicalBeginIntegrationVisitor beginVisitor(this->params, dt);
        node->execute(&beginVisitor);

        sofa::core::MechanicalParams m_mparams(*this->params);
        m_mparams.setDt(dt);

        {
            unsigned int constraintId=0;
            core::ConstraintParams cparams;
            simulation::MechanicalAccumulateConstraint(&cparams, core::MatrixDerivId::constraintJacobian(),constraintId).execute(node);
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
        MechanicalPropagateOnlyPositionAndVelocityVisitor(&m_mparams, nextTime,VecCoordId::position(),VecDerivId::velocity(),
#ifdef SOFA_SUPPORT_MAPPED_MASS
                VecDerivId::dx(),
#endif
                true).execute( node );

        MechanicalEndIntegrationVisitor endVisitor(this->params, dt);
        node->execute(&endVisitor);

        {
            IntegrateEndEvent evBegin;
            PropagateEventVisitor eventPropagation(this->params, &evBegin);
            eventPropagation.execute(node);
        }

        return RESULT_PRUNE;
    }
    {
        // process InteractionForceFields
        for_each(this, node, node->interactionForceField, &AnimateVisitor::fwdInteractionForceField);
        return RESULT_CONTINUE;
    }
}

} // namespace simulation

} // namespace sofa

