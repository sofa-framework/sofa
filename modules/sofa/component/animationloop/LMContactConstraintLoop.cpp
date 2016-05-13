/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/animationloop/LMContactConstraintLoop.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaConstraint/LMConstraintSolver.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/CollisionVisitor.h>
#include <sofa/simulation/common/CollisionBeginEvent.h>
#include <sofa/simulation/common/CollisionEndEvent.h>
#include <sofa/simulation/common/PropagateEventVisitor.h>
#include <sofa/gpu/cuda/CudaRasterizer.h>
#include <sofa/core/ObjectFactory.h>
#include <math.h>
#include <iostream>


//TODO: hope that CudaRasterizer will derive from a non template class Rasterizer. Need an access to the volume detected

namespace sofa
{

namespace component
{

namespace animationloop
{

int LMContactConstraintLoopClass = core::RegisterObject("invert the Sofa simulation pipeline: first integration, than collision detection until no more collision is found.")
        .add< LMContactConstraintLoop >()
        ;

SOFA_DECL_CLASS(LMContactConstraintLoop);

LMContactConstraintLoop::LMContactConstraintLoop(simulation::Node* gnode)
    : Inherit(gnode)
    , maxCollisionSteps( initData(&maxCollisionSteps,(unsigned int)1,"maxSteps", "number of collision steps between each frame rendering") )
{
}

LMContactConstraintLoop::~LMContactConstraintLoop()
{
}


void LMContactConstraintLoop::bwdInit()
{
    //  sout << "collision" << sendl;
    computeCollision();
}

bool LMContactConstraintLoop::needPriorStatePropagation()
{
    using core::behavior::BaseLMConstraint;
    bool needPriorPropagation=false;
    {
        helper::vector< BaseLMConstraint* > c;
        this->getContext()->get<BaseLMConstraint>(&c, core::objectmodel::BaseContext::SearchDown);
        for (unsigned int i=0; i<c.size(); ++i)
        {
            if (!c[i]->isCorrectionComputedWithSimulatedDOF())
            {
                needPriorPropagation=true;
                sout << "Propagating the State because of "<< c[i]->getName() << sendl;
                break;
            }
        }
    }
    return needPriorPropagation;
}

void LMContactConstraintLoop::solveConstraints(bool needPropagation)
{
    //  sout << "apply constraints" << sendl;
    simulation::MechanicalExpressJacobianVisitor JacobianVisitor(this->gnode);
    JacobianVisitor.execute(this->gnode);

    helper::vector< constraintset::LMConstraintSolver* > listSolver;
    this->gnode->get<constraintset::LMConstraintSolver>(&listSolver, core::objectmodel::BaseContext::SearchDown);

    helper::vector< bool > constraintActive(listSolver.size(), false);
    for (unsigned int i=0; i<listSolver.size(); ++i)
    {
        if (listSolver[i]->constraintPos.getValue()) constraintActive[i]=true;
        else                                         listSolver[i]->constraintPos.setValue(true);
    }

    core::behavior::BaseMechanicalState::VecId positionState=core::behavior::BaseMechanicalState::VecId::position();
    simulation::MechanicalSolveLMConstraintVisitor solveConstraintsPosition(positionState,needPropagation, false);
    solveConstraintsPosition.execute(this->gnode);

    for (unsigned int i=0; i<listSolver.size(); ++i)
    {
        listSolver[i]->constraintPos.setValue(constraintActive[i]);
    }

    simulation::MechanicalPropagatePositionVisitor propagateState;
    propagateState.ignoreMask=false;
    propagateState.execute(this->gnode);
}

bool LMContactConstraintLoop::isCollisionDetected()
{
    //  sout << "collision" << sendl;
    {
        simulation::CollisionBeginEvent evBegin;
        simulation::PropagateEventVisitor eventPropagation(&evBegin);
        eventPropagation.execute(getContext());
    }
    ((simulation::Node*) getContext())->execute<simulation::CollisionDetectionVisitor>();
    {
        simulation::CollisionEndEvent evEnd;
        simulation::PropagateEventVisitor eventPropagation(&evEnd);
        eventPropagation.execute(getContext());
    }




    gpu::cuda::CudaRasterizer< defaulttype::Vec3dTypes > *rasterizer;
    this->getContext()->get(rasterizer);

    if (!rasterizer) return false;

    //  sout << "intersections : " << rasterizer->getNbPairs() << sendl;
    return (rasterizer->getNbPairs() != 0);
}

void LMContactConstraintLoop::step(const core::ExecParams* params, double dt)
{
    sofa::helper::AdvancedTimer::stepBegin("AnimationStep");
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Step");
#endif

    {
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        this->gnode->execute ( act );
    }


    double startTime = this->gnode->getTime();
    double mechanicalDt = dt/numMechSteps.getValue();
    AnimateVisitor act(params);
    act.setDt ( mechanicalDt );
    BehaviorUpdatePositionVisitor beh(params , this->gnode->getDt());
    for( unsigned i=0; i<numMechSteps.getValue(); i++ )
    {
        this->gnode->execute ( beh );

        const unsigned int maxSteps = maxCollisionSteps.getValue();

        // Then integrate the time step
        //    sout << "integration" << sendl;
        integrate(dt);

        bool propagateState=needPriorStatePropagation();
        for (unsigned int step=0; step<maxSteps; ++step)
        {
            this->gnode->execute<simulation::MechanicalResetConstraintVisitor>();
            this->gnode->execute<simulation::CollisionResetVisitor>();
            if (isCollisionDetected())
            {
                this->gnode->execute<simulation::CollisionResetVisitor>();
                this->gnode->execute<simulation::CollisionResponseVisitor>();
                solveConstraints(propagateState);
            }
            else
            {
                //No collision --> no constraint
                break;
            }
        }

        this->gnode->setTime ( startTime + (i+1)* act.getDt() );
        sofa::simulation::getSimulation()->getVisualRoot()->setTime ( this->gnode->getTime() );
        this->gnode->execute<UpdateSimulationContextVisitor>(params);  // propagate time
        sofa::simulation::getSimulation()->getVisualRoot()->execute<UpdateSimulationContextVisitor>(params);
        nbMechSteps.setValue(nbMechSteps.getValue() + 1);
    }

    {
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        this->gnode->execute ( act );
    }


    sofa::helper::AdvancedTimer::stepBegin("UpdateMapping");
    //Visual Information update: Ray Pick add a MechanicalMapping used as VisualMapping
    this->gnode->execute<UpdateMappingVisitor>(params);
    sofa::helper::AdvancedTimer::step("UpdateMappingEndEvent");
    {
        UpdateMappingEndEvent ev ( dt );
        PropagateEventVisitor act ( params , &ev );
        this->gnode->execute ( act );
    }
    sofa::helper::AdvancedTimer::stepEnd("UpdateMapping");

#ifndef SOFA_NO_UPDATE_BBOX
    sofa::helper::AdvancedTimer::stepBegin("UpdateBBox");
    this->gnode->execute<UpdateBoundingBoxVisitor>(params);
    sofa::helper::AdvancedTimer::stepEnd("UpdateBBox");
#endif
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Step");
#endif
    nbSteps.setValue(nbSteps.getValue() + 1);

    sofa::helper::AdvancedTimer::stepEnd("AnimationStep");
}

} // namespace animationloop

} // namespace component

} // namespace sofa

