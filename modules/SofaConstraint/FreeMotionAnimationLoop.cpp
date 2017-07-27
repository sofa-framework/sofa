/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#include <SofaConstraint/FreeMotionAnimationLoop.h>
#include <sofa/core/visual/VisualParams.h>

#include <SofaConstraint/LCPConstraintSolver.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/VecId.h>

#include <sofa/helper/AdvancedTimer.h>

#include <sofa/simulation/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/SolveVisitor.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/UpdateMappingEndEvent.h>
#include <sofa/simulation/UpdateBoundingBoxVisitor.h>
#include <sofa/simulation/MechanicalVisitor.h>


namespace sofa
{

namespace simulation
{

///// res += constraint forces (== lambda/dt), only for mechanical object linked to a compliance
//class MechanicalAddComplianceForce : public MechanicalVisitor
//{
//    core::MultiVecDerivId res, lambdas;
//    SReal invdt;
//
//
//public:
//    MechanicalAddComplianceForce(const sofa::core::MechanicalParams* mparams, core::MultiVecDerivId res, core::MultiVecDerivId lambdas, SReal dt)
//        : MechanicalVisitor(mparams), res(res), lambdas(lambdas), invdt(1.0 / dt)
//    {
//#ifdef SOFA_DUMP_VISITOR_INFO
//        setReadWriteVectors();
//#endif
//    }
//
//    // reset lambda where there is no compliant FF
//    // these reseted lambdas were previously propagated, but were not computed from the last solve
//    virtual Result fwdMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* mm)
//    {
//        // a compliant FF must be alone, so if there is one, it is the first one of the list.
//        const core::behavior::BaseForceField* ff = NULL;
//
//        if (!node->forceField.empty()) ff = *node->forceField.begin();
//        else if (!node->interactionForceField.empty()) ff = *node->interactionForceField.begin();
//
//        if (!ff || !ff->isCompliance.getValue())
//        {
//            const core::VecDerivId& lambdasid = lambdas.getId(mm);
//            if (!lambdasid.isNull()) // previously allocated
//            {
//                mm->resetForce(this->params, lambdasid);
//            }
//        }
//        return RESULT_CONTINUE;
//    }
//
//    virtual Result fwdMappedMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* mm)
//    {
//        return fwdMechanicalState(node, mm);
//    }
//
//    // pop-up lamdas without modifying f
//    virtual void bwdMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* map)
//    {
//        map->applyJT(this->mparams, lambdas, lambdas);
//    }
//
//    // for all dofs, f += lambda / dt
//    virtual void bwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* mm)
//    {
//        const core::VecDerivId& lambdasid = lambdas.getId(mm);
//        if (!lambdasid.isNull()) // previously allocated
//        {
//            const core::VecDerivId& resid = res.getId(mm);
//
//            mm->vOp(this->params, resid, resid, lambdasid ); // f += lambda / dt
//        }
//    }
//
//    virtual void bwdMappedMechanicalState(simulation::Node* node, core::behavior::BaseMechanicalState* mm)
//    {
//        bwdMechanicalState(node, mm);
//    }
//
//
//    /// Return a class name for this visitor
//    /// Only used for debugging / profiling purposes
//    virtual const char* getClassName() const { return "MechanicalAddComplianceForce"; }
//    virtual std::string getInfos() const
//    {
//        std::string name = std::string("[") + res.getName() + "," + lambdas.getName() + std::string("]");
//        return name;
//    }
//
//    /// Specify whether this action can be parallelized.
//    virtual bool isThreadSafe() const
//    {
//        return true;
//    }
//
//#ifdef SOFA_DUMP_VISITOR_INFO
//    void setReadWriteVectors()
//    {
//        addWriteVector(res);
//        addWriteVector(lambdas);
//    }
//#endif
//};

}

namespace component
{

namespace animationloop
{

using namespace core::behavior;
using namespace sofa::simulation;

FreeMotionAnimationLoop::FreeMotionAnimationLoop(simulation::Node* gnode)
    : Inherit(gnode)
    , d_linearizeMappingsAroundFreeMotion(initData(&d_linearizeMappingsAroundFreeMotion,false,"linearizeMappingsAroundFreeMotion","If true the linearisation (jacobian) used for constraint accumulation and solving\
                                                                                                                                   will be around the freemotion, otherwise the linearisation around the position at\
                                                                                                                                   the beginning of the time step is used."))
    , m_solveVelocityConstraintFirst(initData(&m_solveVelocityConstraintFirst , false, "solveVelocityConstraintFirst", "solve separately velocity constraint violations before position constraint violations"))
    , constraintSolver(NULL)
    , defaultSolver(NULL)
{
}

FreeMotionAnimationLoop::~FreeMotionAnimationLoop()
{
    if (defaultSolver != NULL)
        defaultSolver.reset();
}

void FreeMotionAnimationLoop::parse ( sofa::core::objectmodel::BaseObjectDescription* arg )
{
    this->simulation::CollisionAnimationLoop::parse(arg);

    defaultSolver = sofa::core::objectmodel::New<constraintset::LCPConstraintSolver>();
    defaultSolver->parse(arg);
}


void FreeMotionAnimationLoop::init()
{

    {
    simulation::common::VectorOperations vop(core::ExecParams::defaultInstance(), this->getContext());
    MultiVecDeriv dx(&vop, core::VecDerivId::dx() ); dx.realloc( &vop, true, true );
    MultiVecDeriv df(&vop, core::VecDerivId::dforce() ); df.realloc( &vop, true, true );
    }




    getContext()->get(constraintSolver, core::objectmodel::BaseContext::SearchDown);
    if (constraintSolver == NULL && defaultSolver != NULL)
    {
        serr << "No ConstraintSolver found, using default LCPConstraintSolver" << sendl;
        this->getContext()->addObject(defaultSolver);
        constraintSolver = defaultSolver.get();
        defaultSolver = NULL;
    }
    else
    {
        defaultSolver.reset();
    }
}


void FreeMotionAnimationLoop::step(const sofa::core::ExecParams* params, SReal dt)
{
    if (dt == 0)
        dt = this->gnode->getDt();

    double startTime = this->gnode->getTime();

    simulation::common::VectorOperations vop(params, this->getContext());
    simulation::common::MechanicalOperations mop(params, this->getContext());

    MultiVecCoord pos(&vop, core::VecCoordId::position() );
    MultiVecDeriv vel(&vop, core::VecDerivId::velocity() );
    MultiVecCoord freePos(&vop, core::VecCoordId::freePosition() );
    MultiVecDeriv freeVel(&vop, core::VecDerivId::freeVelocity() );

    core::ConstraintParams cparams(*params);
    cparams.setX(freePos);
    cparams.setV(freeVel);
    cparams.setJ(sofa::core::ConstMatrixDerivId::holonomicC());
    cparams.setDx(constraintSolver->getDx());
    cparams.setLambda(constraintSolver->getLambda());

    {
        MultiVecDeriv dx(&vop, core::VecDerivId::dx() ); dx.realloc( &vop, true, true );
        MultiVecDeriv df(&vop, core::VecDerivId::dforce() ); df.realloc( &vop, true, true );
    }

    // This solver will work in freePosition and freeVelocity vectors.
    // We need to initialize them if it's not already done.
    sofa::helper::AdvancedTimer::stepBegin("MechanicalVInitVisitor");
    simulation::MechanicalVInitVisitor< core::V_COORD >(params, core::VecCoordId::freePosition(), core::ConstVecCoordId::position(), true).execute(this->gnode);
    simulation::MechanicalVInitVisitor< core::V_DERIV >(params, core::VecDerivId::freeVelocity(), core::ConstVecDerivId::velocity(), true).execute(this->gnode);

    sofa::helper::AdvancedTimer::stepEnd("MechanicalVInitVisitor");


#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Step");
#endif

    {
        sofa::helper::AdvancedTimer::stepBegin("AnimateBeginEvent");
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        this->gnode->execute ( act );
        sofa::helper::AdvancedTimer::stepEnd("AnimateBeginEvent");
    }

    BehaviorUpdatePositionVisitor beh(params , dt);

    using helper::system::thread::CTime;
    using sofa::helper::AdvancedTimer;

    double time = 0.0;
    //double timeTotal = 0.0;
    double timeScale = 1000.0 / (double)CTime::getTicksPerSec();

    if (displayTime.getValue())
    {
        time = (double) CTime::getTime();
        //timeTotal = (double) CTime::getTime();
    }

    // Update the BehaviorModels
    // Required to allow the RayPickInteractor interaction
    dmsg_info() << "updatePos called" ;

    AdvancedTimer::stepBegin("UpdatePosition");
    this->gnode->execute(&beh);
    AdvancedTimer::stepEnd("UpdatePosition");

    dmsg_info() << "updatePos performed - beginVisitor called" ;

    simulation::MechanicalBeginIntegrationVisitor beginVisitor(params, dt);
    this->gnode->execute(&beginVisitor);

    dmsg_info() << "beginVisitor performed - SolveVisitor for freeMotion is called" ;

    // Mapping geometric stiffness coming from previous lambda.
    {
        simulation::MechanicalVOpVisitor lambdaMultInvDt(params, cparams.lambda(), sofa::core::ConstMultiVecId::null(), cparams.lambda(), 1.0 / dt);
        lambdaMultInvDt.setMapped(true);
        this->getContext()->executeVisitor(&lambdaMultInvDt);
        simulation::MechanicalComputeGeometricStiffness geometricStiffnessVisitor(&mop.mparams, cparams.lambda());
        this->getContext()->executeVisitor(&geometricStiffnessVisitor);
    }


    // Free Motion
    AdvancedTimer::stepBegin("FreeMotion");
    simulation::SolveVisitor freeMotion(params, dt, true);
    this->gnode->execute(&freeMotion);
    AdvancedTimer::stepEnd("FreeMotion");

    {
        mop.projectResponse(freeVel);
        mop.propagateDx(freeVel, true);
    }
    dmsg_info() << " SolveVisitor for freeMotion performed" ;

    if (displayTime.getValue())
    {
        msg_info() << " >>>>> Begin display FreeMotionAnimationLoop time  " << msgendl
                   <<" Free Motion " << ((double)CTime::getTime() - time) * timeScale << " ms" ;

        time = (double)CTime::getTime();
    }

    // Collision detection and response creation
    AdvancedTimer::stepBegin("Collision");
    computeCollision(params);
    AdvancedTimer::stepEnd  ("Collision");


    if (!d_linearizeMappingsAroundFreeMotion.getValue())
    {
        // call apply() method in each mapping so as to recompute their linearisation around 
        // the position at the beginning of the time step
        mop.propagateX(pos); 
    }

    if (displayTime.getValue())
    {
        sout << " computeCollision " << ((double) CTime::getTime() - time) * timeScale << " ms" << sendl;
        time = (double)CTime::getTime();
    }

    // Solve constraints
    if (constraintSolver)
    {
        AdvancedTimer::stepBegin("ConstraintSolver");

        if (m_solveVelocityConstraintFirst.getValue())
        {
            cparams.setOrder(core::ConstraintParams::VEL);
            constraintSolver->solveConstraint(&cparams, vel);

            MultiVecDeriv dv(&vop, constraintSolver->getDx());
            mop.projectResponse(dv);
            mop.propagateDx(dv,true);

            // x = xfree + dv * dt
            pos.eq(pos, vel, dt);
        }

        AdvancedTimer::stepEnd("ConstraintSolver");

    }

    if ( displayTime.getValue() )
    {
        sout << " contactCorrections " << ((double)CTime::getTime() - time) * timeScale << " ms" <<sendl;
        sout << "<<<<<< End display FreeMotionAnimationLoop time." << sendl;
    }

    simulation::MechanicalEndIntegrationVisitor endVisitor(params, dt);
    this->gnode->execute(&endVisitor);

    mop.projectPositionAndVelocity(pos, vel);
    mop.propagateXAndV(pos, vel);

    this->gnode->setTime ( startTime + dt );
    this->gnode->execute<UpdateSimulationContextVisitor>(params);  // propagate time

    {
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        this->gnode->execute ( act );
    }


    sofa::helper::AdvancedTimer::stepBegin("UpdateMapping");
    //Visual Information update: Ray Pick add a MechanicalMapping used as VisualMapping
    this->gnode->execute<UpdateMappingVisitor>(params);
//	sofa::helper::AdvancedTimer::step("UpdateMappingEndEvent");
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

}


SOFA_DECL_CLASS(FreeMotionAnimationLoop)

int FreeMotionAnimationLoopClass = core::RegisterObject("Constraint solver")
        .add< FreeMotionAnimationLoop >()
        .addAlias("FreeMotionMasterSolver")
        ;

} // namespace animationloop

} // namespace component

} // namespace sofa
