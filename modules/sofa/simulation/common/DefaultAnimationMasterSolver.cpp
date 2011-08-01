/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/simulation/common/DefaultAnimationMasterSolver.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/common/PrintVisitor.h>
#include <sofa/simulation/common/FindByTypeVisitor.h>
#include <sofa/simulation/common/ExportGnuplotVisitor.h>
#include <sofa/simulation/common/InitVisitor.h>
#include <sofa/simulation/common/InstrumentVisitor.h>
#include <sofa/simulation/common/AnimateVisitor.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/CollisionVisitor.h>
#include <sofa/simulation/common/UpdateContextVisitor.h>
#include <sofa/simulation/common/UpdateMappingVisitor.h>
#include <sofa/simulation/common/ResetVisitor.h>
#include <sofa/simulation/common/VisualVisitor.h>
#include <sofa/simulation/common/ExportOBJVisitor.h>
#include <sofa/simulation/common/WriteStateVisitor.h>
#include <sofa/simulation/common/XMLPrintVisitor.h>
#include <sofa/simulation/common/PropagateEventVisitor.h>
#include <sofa/simulation/common/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/simulation/common/UpdateMappingEndEvent.h>
#include <sofa/simulation/common/CleanupVisitor.h>
#include <sofa/simulation/common/DeleteVisitor.h>
#include <sofa/simulation/common/UpdateBoundingBoxVisitor.h>
#include <sofa/simulation/common/xml/NodeElement.h>

#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/PipeProcess.h>
#include <sofa/helper/AdvancedTimer.h>

#include <sofa/core/visual/VisualParams.h>

#include <stdlib.h>
#include <math.h>


namespace sofa
{

namespace simulation
{

SOFA_DECL_CLASS(DefaultAnimationMasterSolver);

int DefaultAnimationMasterSolverClass = core::RegisterObject("The simplest master solver, created by default when user do not put on scene")
        .add< DefaultAnimationMasterSolver >()
        ;



DefaultAnimationMasterSolver::DefaultAnimationMasterSolver(simulation::Node* _gnode)
    : Inherit()
    , numMechSteps( initData(&numMechSteps,(unsigned) 1,"numMechSteps","Number of mechanical steps within one update step. If the update time step is dt, the mechanical time step is dt/numMechSteps.") )
    , nbSteps( initData(&nbSteps, (unsigned)0, "nbSteps", "Number of animation steps completed", true, false))
    , nbMechSteps( initData(&nbMechSteps, (unsigned)0, "nbMechSteps", "Number of mechanical steps completed", true, false))
    , gnode(_gnode)
{
    std::cout<<" Calling Constructor DefaultAnimationMasterSolver"<<std::endl;
    assert(gnode);
}

DefaultAnimationMasterSolver::~DefaultAnimationMasterSolver()
{

}

void DefaultAnimationMasterSolver::step(const core::ExecParams* params, double dt)
{
    std::cout<<" DefaultAnimationMasterSolver detected on scene and step is called with   dt = "<<dt<<std::endl;


    sofa::helper::AdvancedTimer::begin("Animate");

    {
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        gnode->execute ( act );
    }

    //std::cout << "animate\n";
    double startTime = gnode->getTime();
    double mechanicalDt = dt/numMechSteps.getValue();
    //double nextTime = gnode->getTime() + gnode->getDt();

    // CHANGE to support MasterSolvers : CollisionVisitor is now activated within AnimateVisitor
    //gnode->execute<CollisionVisitor>(params);

    AnimateVisitor act(params);
    act.setDt ( mechanicalDt );
    BehaviorUpdatePositionVisitor beh(params , gnode->getDt());
    for( unsigned i=0; i<numMechSteps.getValue(); i++ )
    {
        gnode->execute ( beh );
        gnode->execute ( act );
        gnode->setTime ( startTime + (i+1)* act.getDt() );
        simulation::getSimulation()->getVisualRoot()->setTime ( gnode->getTime() );
        gnode->execute<UpdateSimulationContextVisitor>(params);  // propagate time
        simulation::getSimulation()->getVisualRoot()->execute<UpdateSimulationContextVisitor>(params);
        nbMechSteps.setValue(nbMechSteps.getValue() + 1);
    }

    {
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        gnode->execute ( act );
    }

    sofa::helper::AdvancedTimer::stepBegin("UpdateMapping");
    //Visual Information update: Ray Pick add a MechanicalMapping used as VisualMapping
    gnode->execute<UpdateMappingVisitor>(params);
    sofa::helper::AdvancedTimer::step("UpdateMappingEndEvent");
    {
        UpdateMappingEndEvent ev ( dt );
        PropagateEventVisitor act ( params , &ev );
        gnode->execute ( act );
    }
    sofa::helper::AdvancedTimer::stepEnd("UpdateMapping");

#ifndef SOFA_NO_UPDATE_BBOX
    sofa::helper::AdvancedTimer::stepBegin("UpdateBBox");
    gnode->execute<UpdateBoundingBoxVisitor>(params);
    sofa::helper::AdvancedTimer::stepEnd("UpdateBBox");
#endif
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode(std::string("Step"));
#endif
    nbSteps.setValue(nbSteps.getValue() + 1);

    sofa::helper::AdvancedTimer::end("Animate");

}

const DefaultAnimationMasterSolver::Solvers& DefaultAnimationMasterSolver::getSolverSequence()
{
    return gnode->solver;
}



} // namespace simulation

} // namespace sofa
