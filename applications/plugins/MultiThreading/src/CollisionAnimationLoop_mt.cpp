/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "CollisionAnimationLoop_mt.h"
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/CollisionVisitor.h>

#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/CollisionBeginEvent.h>
#include <sofa/simulation/CollisionEndEvent.h>
#include <sofa/simulation/IntegrateBeginEvent.h>
#include <sofa/simulation/IntegrateEndEvent.h>

#include <stdlib.h>
#include <math.h>



namespace sofa
{

namespace simulation
{

	using namespace sofa;


CollisionAnimationLoop_mt::CollisionAnimationLoop_mt(simulation::Node* _gnode)
    : Inherit()
    , gnode(_gnode)
{}

CollisionAnimationLoop_mt::~CollisionAnimationLoop_mt()
{}

void CollisionAnimationLoop_mt::collisionReset(const core::ExecParams* params)
{
    if (this->f_printLog.getValue()) 
		std::cerr<<"CollisionAnimationLoop_mt::collisionReset()"<<endl;


    {
        CollisionBeginEvent evBegin;
        PropagateEventVisitor eventPropagation( params /* PARAMS FIRST */, &evBegin);
        eventPropagation.execute(getContext());
    }

	CollisionResetVisitor resetV(params);
    resetV.setTags(this->getTags());
    resetV.execute( getContext() );

}

void CollisionAnimationLoop_mt::collisionCompute(const core::ExecParams* params)
{
    if (this->f_printLog.getValue()) std::cerr<<"CollisionAnimationLoop_mt::collisionCompute()"<<endl;

    CollisionDetectionVisitor detectionV(params);
    detectionV.setTags(this->getTags());
    detectionV.execute( getContext() );

}

void CollisionAnimationLoop_mt::collisionResponse(const core::ExecParams* params)
{
    if (this->f_printLog.getValue()) std::cerr<<"CollisionAnimationLoop_mt::collisionResponse()"<<endl;


	CollisionResponseVisitor responseV(params);
    responseV.setTags(this->getTags());
    responseV.execute( getContext() );

    {
        CollisionEndEvent evEnd;
        PropagateEventVisitor eventPropagation( params /* PARAMS FIRST */, &evEnd);
        eventPropagation.execute(getContext());
    }

}


void CollisionAnimationLoop_mt::integrate(const core::ExecParams* params /* PARAMS FIRST */, double dt)
{

    {
        IntegrateBeginEvent evBegin;
        PropagateEventVisitor eventPropagation( params /* PARAMS FIRST */, &evBegin);
        eventPropagation.execute(getContext());
    }

    MechanicalIntegrationVisitor act( params /* PARAMS FIRST */, dt );
    act.setTags(this->getTags());
    act.execute( getContext() );

    {
        IntegrateEndEvent evBegin;
        PropagateEventVisitor eventPropagation( params /* PARAMS FIRST */, &evBegin);
        eventPropagation.execute(getContext());
    }
}

const CollisionAnimationLoop_mt::Solvers& CollisionAnimationLoop_mt::getSolverSequence()
{
    simulation::Node* gnode = dynamic_cast<simulation::Node*>( getContext() );
    assert( gnode );
    return gnode->solver;
}

// CollisionAnimationLoop_mt::Pipeline* CollisionAnimationLoop_mt::getPipeline()
// {
// 	simulation::Node* gnode = dynamic_cast<simulation::Node*>( getContext() );
// 	assert( gnode );
// 	return gnode->collisionPipeline;
// }


} // namespace simulation

} // namespace sofa
