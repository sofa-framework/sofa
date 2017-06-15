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
#include <sofa/simulation/CollisionAnimationLoop.h>

#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/AnimateVisitor.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/CollisionVisitor.h>
#include <sofa/simulation/CollisionBeginEvent.h>
#include <sofa/simulation/CollisionEndEvent.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/IntegrateBeginEvent.h>
#include <sofa/simulation/IntegrateEndEvent.h>
#include <sofa/simulation/UpdateMappingEndEvent.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/UpdateBoundingBoxVisitor.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/BehaviorUpdatePositionVisitor.h>


#include <stdlib.h>
#include <math.h>



namespace sofa
{

namespace simulation
{


CollisionAnimationLoop::CollisionAnimationLoop(simulation::Node* _gnode)
    : Inherit()
    , gnode(_gnode)
{}

CollisionAnimationLoop::~CollisionAnimationLoop()
{}

void CollisionAnimationLoop::computeCollision(const core::ExecParams* params)
{
    dmsg_info() <<"CollisionAnimationLoop::computeCollision()" ;

    {
        CollisionBeginEvent evBegin;
        PropagateEventVisitor eventPropagation( params, &evBegin);
        eventPropagation.execute(getContext());
    }

    CollisionVisitor act(params);
    act.setTags(this->getTags());
    act.execute( getContext() );

    {
        CollisionEndEvent evEnd;
        PropagateEventVisitor eventPropagation( params, &evEnd);
        eventPropagation.execute(getContext());
    }
}

void CollisionAnimationLoop::integrate(const core::ExecParams* params, SReal dt)
{

    {
        IntegrateBeginEvent evBegin;
        PropagateEventVisitor eventPropagation( params, &evBegin);
        eventPropagation.execute(getContext());
    }

    MechanicalIntegrationVisitor act( params, dt );
    act.setTags(this->getTags());
    act.execute( getContext() );

    {
        IntegrateEndEvent evBegin;
        PropagateEventVisitor eventPropagation( params, &evBegin);
        eventPropagation.execute(getContext());
    }
}

const CollisionAnimationLoop::Solvers& CollisionAnimationLoop::getSolverSequence()
{
    simulation::Node* gnode = dynamic_cast<simulation::Node*>( getContext() );
    assert( gnode );
    return gnode->solver;
}

// CollisionAnimationLoop::Pipeline* CollisionAnimationLoop::getPipeline()
// {
// 	simulation::Node* gnode = dynamic_cast<simulation::Node*>( getContext() );
// 	assert( gnode );
// 	return gnode->collisionPipeline;
// }


} // namespace simulation

} // namespace sofa
