#include"CompliantPostStabilizationAnimationLoop.h"
#include <sofa/core/ObjectFactory.h>
#include <Compliant/odesolver/CompliantImplicitSolver.h>
#include <SofaBaseCollision/DefaultContactManager.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/UpdateMappingEndEvent.h>
#include <sofa/simulation/UpdateBoundingBoxVisitor.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/BehaviorUpdatePositionVisitor.h>

using namespace sofa::core::objectmodel;
using namespace sofa::core::behavior;
using namespace sofa::simulation;


namespace sofa {

using namespace core::objectmodel;
using namespace core::behavior;
using namespace simulation;


namespace component {

namespace animationloop {



CompliantPostStabilizationAnimationLoop::CompliantPostStabilizationAnimationLoop(simulation::Node* gnode)
    : Inherit(gnode)
    , m_solver(NULL)
    , m_contact(NULL)
{}



void CompliantPostStabilizationAnimationLoop::init()
{
    Inherit::init();

    getContext()->get(m_solver, core::objectmodel::BaseContext::SearchDown);
    if( !m_solver ) serr<<"must be used with a CompliantImplicitSolver\n";

    getContext()->get(m_contact, core::objectmodel::BaseContext::SearchDown);
    if( !m_contact ) serr<<"must be used with a DefaultContactManager";

    m_responseId = m_contact->response.getValue().getSelectedId();
    m_responseParams = m_contact->responseParams.getValue();

    m_correctionResponseId = m_contact->response.getValue().isInOptionsList( "CompliantContact" );
    m_correctionResponseParams = "damping=0&compliance=0&holonomic=1";
}


void CompliantPostStabilizationAnimationLoop::step(const sofa::core::ExecParams* params, SReal dt)
{
    // TODO handle dt as in defaultanimationloop
    if (dt == 0) dt = this->gnode->getDt();


    // the stabilization will be handled by this animation loop, so enforce no stabilization during solve
    if( m_solver->stabilization.getValue().getSelectedId()!=odesolver::CompliantImplicitSolver::NO_STABILIZATION )
    {
        m_solver->stabilization.beginWriteOnly()->setSelectedItem(odesolver::CompliantImplicitSolver::NO_STABILIZATION);
        m_solver->stabilization.endEdit();
    }




    {
        simulation::AnimateBeginEvent ev ( dt );
        simulation::PropagateEventVisitor act ( params, &ev );
        this->gnode->execute ( act );
    }

    SReal startTime = this->gnode->getTime();

    odesolver::CompliantImplicitSolver::SolverOperations sop( params, m_solver->getContext(), m_solver->alpha.getValue(), m_solver->beta.getValue(), dt, core::VecCoordId::position(), core::VecDerivId::velocity() );


    MultiVecCoord pos(&sop.vop, core::VecCoordId::position() );
    MultiVecDeriv vel(&sop.vop, core::VecDerivId::velocity() );

    BehaviorUpdatePositionVisitor beh(params , dt);
    this->gnode->execute(&beh);


    // compute collision using selected method (eg with friction) Holonomic contact should logically be used
    computeCollision();


    simulation::MechanicalBeginIntegrationVisitor beginVisitor(params, dt);
    this->gnode->execute(&beginVisitor);

    // solve the system with full contact
    m_solver->solve(params,dt,pos,vel);

    sop.mop.propagateXAndV(pos,vel);


    // replace the current ContactManager response by the one creating unilateral contacts for correction pass
    m_contact->response.beginWriteOnly()->setSelectedItem(m_correctionResponseId); m_contact->response.endEdit();
    m_contact->responseParams.setValue(m_correctionResponseParams);

    // create UNILATERAL CONTACTS with the new positions
    computeCollision(params);

    // restore the previous ContactManager response
    m_contact->response.beginWriteOnly()->setSelectedItem(m_responseId); m_contact->response.endEdit();
    m_contact->responseParams.setValue(m_responseParams);

    // solve the correction system
    m_solver->post_stabilization( sop, pos.id(), vel.id(), true, true );

    simulation::MechanicalEndIntegrationVisitor endVisitor(params, dt);
    this->gnode->execute(&endVisitor);

    this->gnode->setTime ( startTime + dt );
    this->gnode->execute<simulation::UpdateSimulationContextVisitor>(params);  // propagate time

    {
        simulation::AnimateEndEvent ev ( dt );
        simulation::PropagateEventVisitor act ( params, &ev );
        this->gnode->execute ( act );
    }

    //Visual Information update: Ray Pick add a MechanicalMapping used as VisualMapping
    this->gnode->execute<simulation::UpdateMappingVisitor>(params);
    {
        simulation::UpdateMappingEndEvent ev ( dt );
        simulation::PropagateEventVisitor act ( params , &ev );
        this->gnode->execute ( act );
    }

#ifndef SOFA_NO_UPDATE_BBOX
    this->gnode->execute<simulation::UpdateBoundingBoxVisitor>(params);
#endif

}







SOFA_DECL_CLASS(CompliantPostStabilizationAnimationLoop)


int CompliantPostStabilizationAnimationLoopClass = core::RegisterObject("CompliantPostStabilizationAnimationLoop").add< CompliantPostStabilizationAnimationLoop >();

} // namespace animationloop

} // namespace component

} // namespace sofa
