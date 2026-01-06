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
#include <sofa/component/animationloop/ConstraintAnimationLoop.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/component/constraint/lagrangian/solver/ConstraintSolverImpl.h>
#include <sofa/core/behavior/ConstraintResolution.h>

#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/SolveVisitor.h>
#include <sofa/simulation/UpdateInternalDataVisitor.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/UpdateMappingEndEvent.h>
#include <sofa/simulation/UpdateBoundingBoxVisitor.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/helper/LCPcalc.h>
#include <sofa/helper/ScopedAdvancedTimer.h>


#include <sofa/core/ObjectFactory.h>

#include <sofa/core/behavior/BaseLagrangianConstraint.h> ///< ConstraintResolution.

#include <sofa/helper/AdvancedTimer.h>

#include <thread>

#include <sofa/simulation/mechanicalvisitor/MechanicalVInitVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalVInitVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalBeginIntegrationVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalBeginIntegrationVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalVOpVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalVOpVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalProjectPositionVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalProjectPositionVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyPositionVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalPropagateOnlyPositionVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalProjectJacobianMatrixVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalProjectJacobianMatrixVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalEndIntegrationVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalEndIntegrationVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalResetConstraintVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalResetConstraintVisitor;

#include <sofa/component/constraint/lagrangian/solver/visitors/MechanicalGetConstraintResolutionVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalBuildConstraintMatrix.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalAccumulateMatrixDeriv.h>

/// Change that to true if you want to print extra message on this component.
/// You can eventually link that to an object attribute.
#define EMIT_EXTRA_DEBUG_MESSAGE false

namespace sofa::component::animationloop
{

using namespace sofa::linearalgebra;
using namespace sofa::defaulttype;
using namespace helper::system::thread;
using namespace core::behavior;
using namespace sofa::simulation;

ConstraintProblem::ConstraintProblem(bool printLog)
{
    SOFA_UNUSED(printLog);

    this->_tol = 0.0001;
    this->_dim = 0;

    _timer = new CTime();
}

ConstraintProblem::~ConstraintProblem()
{
    _dFree.clear();
    _d.clear();
    _W.clear();
    _force.clear();
    // if not null delete the old constraintProblem
    for(int i=0; i<_dim; i++)
    {
        if (_constraintsResolutions[i] != nullptr)
        {
            delete _constraintsResolutions[i];
            _constraintsResolutions[i] = nullptr;
        }
    }
    _constraintsResolutions.clear(); // _constraintsResolutions.clear();
    delete(_timer);
}

void ConstraintProblem::clear(int dim, const SReal&tol)
{
    // if not null delete the old constraintProblem
    for(int i=0; i<_dim; i++)
    {
        if (_constraintsResolutions[i] != nullptr)
        {
            delete _constraintsResolutions[i];
            _constraintsResolutions[i] = nullptr;
        }
    }
    _dFree.clear();
    _dFree.resize(dim);
    _d.resize(dim);
    _W.resize(dim,dim);
    _force.resize(dim);
    _df.resize(dim);
    _constraintsResolutions.resize(dim); // _constraintsResolutions.clear();
    this->_tol = tol;
    this->_dim = dim;
}


void ConstraintProblem::gaussSeidelConstraintTimed(SReal &timeout, int numItMax)
{
    SReal error=0.0;

    const SReal t0 = (SReal)_timer->getTime() ;
    const SReal timeScale = 1.0 / (SReal)CTime::getTicksPerSec();

    for(int i=0; i<numItMax; i++)
    {
        error=0.0;
        for(int j=0; j<_dim; ) // increment of j realized at the end of the loop
        {
            const int nb = _constraintsResolutions[j]->getNbLines();

            //2. for each line we compute the actual value of d
            //   (a)d is set to dfree
            std::vector<SReal> errF(&_force[j], &_force[j+nb]);
            std::copy_n(_dFree.begin() + j, nb, _d.begin() + j);

            //   (b) contribution of forces are added to d
            for(int k=0; k<_dim; k++)
                for(int l=0; l<nb; l++)
                    _d[j+l] += _W(j+l,k) * _force[k];



            //3. the specific resolution of the constraint(s) is called
            //SReal** w = this->_W.ptr();
            _constraintsResolutions[j]->resolution(j, this->getW()->lptr(), this->getD()->ptr(), this->getF()->ptr(), _dFree.ptr());

            //4. the error is measured (displacement due to the new resolution (i.e. due to the new force))
            if(nb > 1)
            {
                SReal terr = 0.0;
                for(int l=0; l<nb; l++)
                {
                    SReal terr2 = 0;
                    for (int m=0; m<nb; m++)
                    {
                        terr2 += _W(j+l,j+m) * (_force[j+m] - errF[m]);
                    }
                    terr += terr2 * terr2;
                }
                error += sqrt(terr);
            }
            else
                error += fabs(_W(j,j) * (_force[j] - errF[0]));

            j += nb;
        }

        /////////////////// GAUSS SEIDEL IS TIMED !!! /////////
        const SReal t1 = (SReal)_timer->getTime();
        const SReal dt = (t1 - t0)*timeScale;
        if(dt > timeout)
        {
            return;
        }
        ///////////////////////////////////////////////////////

        if(error < _tol*(_dim+1) && i>0) // do not stop at the first iteration (that is used for initial guess computation)
        {
            return;
        }
    }

    msg_info("ConstraintAnimationLoop") << "------  No convergence in gaussSeidelConstraint Timed before time criterion !: error = "
               << error << " ------" << msgendl;

}

ConstraintAnimationLoop::ConstraintAnimationLoop() :
    d_displayTime(initData(&d_displayTime, false, "displayTime","Display time for each important step of ConstraintAnimationLoop."))
    , d_tol( initData(&d_tol, 0.00001_sreal, "tolerance", "Tolerance of the Gauss-Seidel"))
    , d_maxIt( initData(&d_maxIt, 1000, "maxIterations", "Maximum number of iterations of the Gauss-Seidel"))
    , d_doCollisionsFirst(initData(&d_doCollisionsFirst, false, "doCollisionsFirst","Compute the collisions first (to support penality-based contacts)"))
    , d_doubleBuffer( initData(&d_doubleBuffer, false, "doubleBuffer","Double the buffer dedicated to the constraint problem to make it accessible to another thread"))
    , d_scaleTolerance( initData(&d_scaleTolerance, true, "scaleTolerance","Scale the error tolerance with the number of constraints"))
    , d_allVerified( initData(&d_allVerified, false, "allVerified","All constraints must be verified (each constraint's error < tolerance)"))
    , d_sor( initData(&d_sor, 1.0_sreal, "sor","Successive Over Relaxation parameter (0-2)"))
    , d_schemeCorrection( initData(&d_schemeCorrection, false, "schemeCorrection","Apply new scheme where compliance is progressively corrected"))
    , d_realTimeCompensation( initData(&d_realTimeCompensation, false, "realTimeCompensation","If the total computational time T < dt, sleep(dt-T)"))
    , d_graphErrors( initData(&d_graphErrors,"graphErrors","Sum of the constraints' errors at each iteration"))
    , d_graphConstraints( initData(&d_graphConstraints,"graphConstraints","Graph of each constraint's error at the end of the resolution"))
    , d_graphForces( initData(&d_graphForces,"graphForces","Graph of each constraint's force at each step of the resolution"))
{
    bufCP1 = false;

    d_graphErrors.setWidget("graph");
    d_graphErrors.setGroup("Graph");

    d_graphConstraints.setWidget("graph");
    d_graphConstraints.setGroup("Graph");

    d_graphForces.setWidget("graph");
    d_graphForces.setGroup("Graph2");

    CP1.clear(0,d_tol.getValue());
    CP2.clear(0,d_tol.getValue());

    timer = nullptr;

    msg_deprecated("ConstraintAnimationLoop") << "WARNING : ConstraintAnimationLoop is deprecated. Please use the combination of FreeMotionAnimationLoop and GenericConstraintSolver." ;
}

ConstraintAnimationLoop::~ConstraintAnimationLoop()
{
    if (timer != nullptr)
    {
        delete timer;
        timer = nullptr;
    }
}

void ConstraintAnimationLoop::init()
{
    // Prevents ConstraintCorrection accumulation due to multiple AnimationLoop initialization on dynamic components Add/Remove operations.
    if (!constraintCorrections.empty())
    {
        constraintCorrections.clear();
    }

    getContext()->get<core::behavior::BaseConstraintCorrection> ( &constraintCorrections, core::objectmodel::BaseContext::SearchDown );
}


void ConstraintAnimationLoop::launchCollisionDetection(const core::ExecParams* params)
{
    dmsg_info_when(EMIT_EXTRA_DEBUG_MESSAGE)
            <<"computeCollision is called";

    ////////////////// COLLISION DETECTION///////////////////////////////////////////////////////////////////////////////////////////
    {
        SCOPED_TIMER("Collision");
        computeCollision(params);
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if ( d_displayTime.getValue() )
    {
        msg_info() <<" computeCollision                 " << ( (SReal) timer->getTime() - time)*timeScale <<" ms";
        time = (SReal) timer->getTime();
    }

}


void ConstraintAnimationLoop::freeMotion(const core::ExecParams* params, simulation::Node *context, SReal &dt)
{
    dmsg_info_when(EMIT_EXTRA_DEBUG_MESSAGE)
            <<"Free Motion is called" ;

    ///////////////////////////////////////////// FREE MOTION /////////////////////////////////////////////////////////////
    {
        SCOPED_TIMER_VARNAME(freeMotionTimer, "Free Motion");

        MechanicalBeginIntegrationVisitor(params, dt).execute(context);

        ////////////////// (optional) PREDICTIVE CONSTRAINT FORCES ///////////////////////////////////////////////////////////////////////////////////////////
        /// When scheme Correction is used, the constraint forces computed at the previous time-step
        /// are applied during the first motion, so which is no more a "free" motion but a "predictive" motion
        ///////////
        if(d_schemeCorrection.getValue())
        {
            sofa::core::ConstraintParams cparams(*params);
            sofa::core::MultiVecDerivId f =  core::vec_id::write_access::externalForce;

            for (auto cc : constraintCorrections)
            {
                cc->applyPredictiveConstraintForce(&cparams, f, getCP()->getF());
            }
        }

        simulation::SolveVisitor(params, dt, true).execute(context);

        {
            sofa::core::MechanicalParams mparams(*params);
            sofa::core::MultiVecCoordId xfree = sofa::core::vec_id::write_access::freePosition;
            mparams.x() = xfree;
            MechanicalProjectPositionVisitor(&mparams, 0, xfree ).execute(context);
            MechanicalPropagateOnlyPositionVisitor(&mparams, 0, xfree ).execute(context);
        }
    }

    //////// TODO : propagate velocity !!

    ////////propagate acceleration ? //////

    //this is done to set dx to zero in subgraph
    core::MultiVecDerivId dx_id = core::vec_id::write_access::dx;
    MechanicalVOpVisitor(params, dx_id, core::ConstVecId::null(), core::ConstVecId::null(), 1.0 ).setMapped(true).execute(context);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if ( d_displayTime.getValue() )
    {
        msg_info() << ">>>>> Begin display ConstraintAnimationLoop time";
        msg_info() <<" Free Motion                           " << ( (SReal) timer->getTime() - time)*timeScale <<" ms";
        time = (SReal) timer->getTime();
    }
}

void ConstraintAnimationLoop::setConstraintEquations(const core::ExecParams* params, simulation::Node *context)
{
    for (const auto cc : constraintCorrections)
    {
        cc->resetContactForce();
    }

    //////////////////////////////////////CONSTRAINTS RESOLUTION//////////////////////////////////////////////////////////////////////
    msg_info_when(EMIT_EXTRA_DEBUG_MESSAGE) <<"constraints Matrix construction is called" ;

    {
        SCOPED_TIMER_VARNAME(constraintDefinitionTimer, "Constraints definition");

        if(!d_schemeCorrection.getValue())
        {
            /// calling resetConstraint & setConstraint & accumulateConstraint visitors
            /// and resize the constraint problem that will be solved
            unsigned int numConstraints = 0;
            writeAndAccumulateAndCountConstraintDirections(params, context, numConstraints);
        }


        core::MechanicalParams mparams = core::MechanicalParams(*params);
        MechanicalProjectJacobianMatrixVisitor(&mparams).execute(context);

        /// calling GetConstraintViolationVisitor: each constraint provides its present violation
        /// for a given state (by default: free_position TODO: add VecId to make this method more generic)
        getIndividualConstraintViolations(params, context);

        if(!d_schemeCorrection.getValue())
        {
            /// calling getConstraintResolution: each constraint provides a method that is used to solve it during GS iterations
            getIndividualConstraintSolvingProcess(params, context);
        }
    }

    /// calling getCompliance projected in the contact space => getDelassusOperator(_W) = H*C*Ht
    computeComplianceInConstraintSpace();

    if ( d_displayTime.getValue() )
    {
        msg_info()<<" Build problem in the constraint space " << ( (SReal) timer->getTime() - time)*timeScale<<" ms";
        time = (SReal) timer->getTime();
    }
}

void ConstraintAnimationLoop::writeAndAccumulateAndCountConstraintDirections(const core::ExecParams* params, simulation::Node *context, unsigned int &numConstraints)
{
    core::ConstraintParams cparams = core::ConstraintParams(*params);
    cparams.setX(core::vec_id::read_access::freePosition);
    cparams.setV(core::vec_id::read_access::freeVelocity);

    // calling resetConstraint on LMConstraints and MechanicalStates
    MechanicalResetConstraintVisitor(&cparams).execute(context);

    // calling applyConstraint on each constraint
    sofa::simulation::mechanicalvisitor::MechanicalBuildConstraintMatrix(&cparams, core::vec_id::write_access::constraintJacobian, numConstraints).execute(context);

    sofa::helper::AdvancedTimer::valSet("numConstraints", numConstraints);

    // calling accumulateConstraint on the mappings
    sofa::simulation::mechanicalvisitor::MechanicalAccumulateMatrixDeriv(&cparams, core::vec_id::write_access::constraintJacobian).execute(context);

    getCP()->clear(numConstraints,this->d_tol.getValue());
}

void ConstraintAnimationLoop::getIndividualConstraintViolations(const core::ExecParams* params, simulation::Node *context)
{
    core::ConstraintParams cparams = core::ConstraintParams(*params);
    cparams.setX(core::vec_id::read_access::freePosition);
    cparams.setV(core::vec_id::read_access::freeVelocity);

    constraint::lagrangian::solver::MechanicalGetConstraintViolationVisitor(&cparams, getCP()->getDfree()).execute(context);
}

void ConstraintAnimationLoop::getIndividualConstraintSolvingProcess(const core::ExecParams* params, simulation::Node *context)
{
    /// calling getConstraintResolution: each constraint provides a method that is used to solve it during GS iterations
    core::ConstraintParams cparams = core::ConstraintParams(*params);
    cparams.setX(core::vec_id::read_access::freePosition);
    cparams.setV(core::vec_id::read_access::freeVelocity);

    sofa::component::constraint::lagrangian::solver::MechanicalGetConstraintResolutionVisitor(&cparams, getCP()->getConstraintResolutions(), 0).execute(context);
}

void ConstraintAnimationLoop::computeComplianceInConstraintSpace()
{
    /// calling getCompliance => getDelassusOperator(_W) = H*C*Ht
    dmsg_info_when(EMIT_EXTRA_DEBUG_MESSAGE) << "   4. get Compliance " ;

    SCOPED_TIMER_VARNAME(getComplianceTimer, "Get Compliance");
    for (const auto cc : constraintCorrections)
    {
        cc->addComplianceInConstraintSpace(core::constraintparams::defaultInstance(), getCP()->getW());
    }
}

void ConstraintAnimationLoop::correctiveMotion(const core::ExecParams* params, simulation::Node *node)
{
    dmsg_info_when(EMIT_EXTRA_DEBUG_MESSAGE)
            <<"constraintCorrections motion is called" ;

    SCOPED_TIMER_VARNAME(correctiveMotionTimer, "Corrective Motion");

    if(d_schemeCorrection.getValue())
    {
        // IF SCHEME CORRECTIVE=> correct the motion using dF
        for (const auto cc : constraintCorrections)
        {
            cc->applyContactForce(getCP()->getdF());
        }
    }
    else
    {
        // ELSE => only correct the motion using F
        for (const auto cc : constraintCorrections)
        {
            cc->applyContactForce(getCP()->getF());
        }
    }

    simulation::common::MechanicalOperations mop(params, node);

    mop.propagateV(core::vec_id::write_access::velocity);

    mop.propagateDx(core::vec_id::write_access::dx, true);

    // "mapped" x = xfree + dx
    MechanicalVOpVisitor(params, core::vec_id::write_access::position, core::vec_id::read_access::freePosition, core::vec_id::read_access::dx, 1.0 ).setOnlyMapped(true).execute(node);

    if(!d_schemeCorrection.getValue())
    {
        for (const auto cc : constraintCorrections)
        {
            cc->resetContactForce();
        }
    }
}

void ConstraintAnimationLoop::step ( const core::ExecParams* params, SReal dt )
{
    auto node = dynamic_cast<sofa::simulation::Node*>(this->l_node.get());

    static SReal simulationTime=0.0;

    simulationTime+=dt;
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Step");
#endif

    {
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        node->execute ( act );
    }
    
    
    SReal startTime = node->getTime();
    
    BehaviorUpdatePositionVisitor beh(params , node->getDt());
    node->execute ( beh );

    UpdateInternalDataVisitor uid(params);
    node->execute ( uid );


    if (simulationTime>0.1)
        d_activateSubGraph.setValue(true);
    else
        d_activateSubGraph.setValue(false);

    time = 0.0;
    SReal totaltime = 0.0;
    timeScale = 1.0 / (SReal)CTime::getTicksPerSec() * 1000;
    if ( d_displayTime.getValue() )
    {
        if (timer == nullptr)
            timer = new CTime();

        time = (SReal) timer->getTime();
        totaltime = time;
        msg_info()<<msgendl;
    }
    if (d_doubleBuffer.getValue())
    {
        // SWAP BUFFER:
        bufCP1 = !bufCP1;
    }

    ConstraintProblem& CP = (d_doubleBuffer.getValue() && bufCP1) ? CP2 : CP1;

#if !defined(WIN32)
    if (d_realTimeCompensation.getValue())
    {
        if (timer == nullptr)
        {
            timer = new CTime();
            compTime = iterationTime = (SReal)timer->getTime();
        }
        else
        {
            SReal actTime = SReal(timer->getTime());
            SReal compTimeDiff = actTime - compTime;
            SReal iterationTimeDiff = actTime - iterationTime;
            iterationTime = actTime;
            msg_info() << "Total time = " << iterationTimeDiff ;
            int toSleep = (int)floor(dt*1000000-compTimeDiff);
            if (toSleep > 0)
                std::this_thread::sleep_for(std::chrono::microseconds(toSleep));
            else
                msg_error() << "Cannot achieve frequency for dt = " << dt ;
            compTime = (SReal)timer->getTime();
        }
    }
#endif

    dmsg_info() << " step is called" ;

    // This solver will work in freePosition and freeVelocity vectors.
    // We need to initialize them if it's not already done.
    MechanicalVInitVisitor<core::V_COORD>(params, core::vec_id::write_access::freePosition, core::vec_id::read_access::position, true).execute(node);
    MechanicalVInitVisitor<core::V_DERIV>(params, core::vec_id::write_access::freeVelocity, core::vec_id::read_access::velocity).execute(node);

    if (d_doCollisionsFirst.getValue())
    {
        /// COLLISION
        launchCollisionDetection(params);
    }

    // Update the BehaviorModels => to be removed ?
    // Required to allow the RayPickInteractor interaction
    {
        SCOPED_TIMER_VARNAME(behaviorUpdateTimer, "BehaviorUpdate");
        simulation::BehaviorUpdatePositionVisitor(params, dt).execute(node);
    }


    if(d_schemeCorrection.getValue())
    {
        // Compute the predictive force:
        numConstraints = 0;

        //1. Find the new constraint direction
        writeAndAccumulateAndCountConstraintDirections(params, node, numConstraints);

        //2. Get the constraint solving process:
        getIndividualConstraintSolvingProcess(params, node);

        //3. Use the stored forces to compute
        if (EMIT_EXTRA_DEBUG_MESSAGE)
        {
            computePredictiveForce(CP.getSize(), CP.getF()->ptr(), CP.getConstraintResolutions());
            msg_info() << "getF() after computePredictiveForce:" ;
            helper::resultToString(std::cout,CP.getF()->ptr(),CP.getSize());
        }
    }

    if (EMIT_EXTRA_DEBUG_MESSAGE)
    {
        (*CP.getF())*=0.0;
        computePredictiveForce(CP.getSize(), CP.getF()->ptr(), CP.getConstraintResolutions());
        msg_info() << "getF() after re-computePredictiveForce:" ;
        helper::resultToString(std::cout,CP.getF()->ptr(),CP.getSize());
    }

    /// FREE MOTION
    freeMotion(params, node, dt);



    if (!d_doCollisionsFirst.getValue())
    {
        /// COLLISION
        launchCollisionDetection(params);
    }

    //////////////// BEFORE APPLYING CONSTRAINT  : propagate position through mapping
    core::MechanicalParams mparams(*params);
    MechanicalProjectPositionVisitor(&mparams, 0, core::vec_id::write_access::position).execute(node);
    MechanicalPropagateOnlyPositionVisitor(&mparams, 0, core::vec_id::write_access::position).execute(node);


    /// CONSTRAINT SPACE & COMPLIANCE COMPUTATION
    setConstraintEquations(params, node);

    if (EMIT_EXTRA_DEBUG_MESSAGE)
    {
        msg_info() << "getF() after setConstraintEquations:" ;
        helper::resultToString(std::cout, CP.getF()->ptr(),CP.getSize());
    }

    {
        SCOPED_TIMER_VARNAME(gaussSeidelTimer, "GaussSeidel");
        if (EMIT_EXTRA_DEBUG_MESSAGE)
            msg_info() << "Gauss-Seidel solver is called on problem of size " << CP.getSize() ;

        if(d_schemeCorrection.getValue())
            (*CP.getF())*=0.0;

        gaussSeidelConstraint(CP.getSize(), CP.getDfree()->ptr(), CP.getW()->lptr(), CP.getF()->ptr(), CP.getD()->ptr(), CP.getConstraintResolutions(), CP.getdF()->ptr());
    }

    if (EMIT_EXTRA_DEBUG_MESSAGE)
        helper::printLCP(CP.getDfree()->ptr(), CP.getW()->lptr(), CP.getF()->ptr(),  CP.getSize());

    if ( d_displayTime.getValue() )
    {
        msg_info() << " Solve with GaussSeidel                " << ( (SReal) timer->getTime() - time)*timeScale<<" ms" ;
        time = (SReal) timer->getTime();
    }

    /// CORRECTIVE MOTION
    correctiveMotion(params, node);

    if ( d_displayTime.getValue() )
    {
        msg_info() << " ContactCorrections                    " << ( (SReal) timer->getTime() - time)*timeScale <<" ms" << msgendl
                   << "  = Total                              " << ( (SReal) timer->getTime() - totaltime)*timeScale <<" ms" << msgendl
                   << " With : " << CP.getSize() << " constraints" << msgendl
                   << "<<<<< End display ConstraintAnimationLoop time." ;
    }

    MechanicalEndIntegrationVisitor endVisitor(params, dt);
    node->execute(&endVisitor);
    node->setTime ( startTime + dt );
    node->execute<UpdateSimulationContextVisitor>(params);  // propagate time

    {
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        node->execute ( act );
    }

    {
        SCOPED_TIMER_VARNAME(updateMappingTimer, "UpdateMapping");

        node->execute<UpdateMappingVisitor>(params);
        sofa::helper::AdvancedTimer::step("UpdateMappingEndEvent");
        {
            UpdateMappingEndEvent ev ( dt );
            PropagateEventVisitor act ( params , &ev );
            node->execute ( act );
        }
    }

    if (d_computeBoundingBox.getValue())
    {
        SCOPED_TIMER_VARNAME(updateBBoxTimer, "UpdateBBox");
        node->execute<UpdateBoundingBoxVisitor>(params);
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Step");
#endif


}

void ConstraintAnimationLoop::computePredictiveForce(int dim, SReal* force, std::vector<core::behavior::ConstraintResolution*>& res)
{
    for(int i=0; i<dim; )
    {
        res[i]->initForce(i, force);
        i += res[i]->getNbLines();
    }
}

void ConstraintAnimationLoop::gaussSeidelConstraint(int dim, SReal* dfree, SReal** w, SReal* force,
        SReal* d, std::vector<ConstraintResolution*>& res, SReal* df=nullptr)
{
    if(!dim)
        return;

    int iter, nb;
    SReal error=0.0;

    SReal tolerance = d_tol.getValue();
    int numItMax = d_maxIt.getValue();
    bool convergence = false;
    SReal sor = d_sor.getValue();
    bool allVerified = d_allVerified.getValue();
    sofa::type::vector<SReal> tempForces;
    if(sor != 1.0) tempForces.resize(dim);

    if(d_scaleTolerance.getValue() && !allVerified)
        tolerance *= dim;

    for(int i=0; i<dim; )
    {
        res[i]->init(i, w, force);
        i += res[i]->getNbLines();
    }

    {
        auto* graphs = d_graphForces.beginEdit();
        graphs->clear();
        d_graphForces.endEdit();
    }

    if(d_schemeCorrection.getValue())
    {
        msg_info() << "shemeCorrection => LCP before step 1";
        helper::printLCP(dfree, w, force,  dim);
        ///////// scheme correction : step 1 => modification of dfree
        for(int j=0; j<dim; j++)
        {
            for(int k=0; k<dim; k++)
                dfree[j] -= w[j][k] * force[k];
        }

        ///////// scheme correction : step 2 => storage of force value
        for(int j=0; j<dim; j++)
            df[j] = -force[j];
    }

    sofa::type::vector<SReal>& graph_residuals = (*d_graphErrors.beginEdit())["Error"];
    graph_residuals.clear();

    sofa::type::vector<SReal> tabErrors;
    tabErrors.resize(dim);

    for(iter=0; iter<numItMax; iter++)
    {
        bool constraintsAreVerified = true;
        if(sor != 1.0)
        {
            std::copy_n(force, dim, tempForces.begin());
        }

        error=0.0;

        for(int j=0; j<dim; ) // increment of j realized at the end of the loop
        {
            //1. nbLines provide the dimension of the constraint
            nb = res[j]->getNbLines();

            bool check = true;
            for (int b=0; b<nb; b++)
            {
                if (w[j+b][j+b]==0.0) check = false;
            }

            if (check)
            {
                //2. for each line we compute the actual value of d
                //   (a)d is set to dfree
                std::vector<SReal> errF(&force[j], &force[j+nb]);
                std::copy_n(&dfree[j], nb, &d[j]);

                //   (b) contribution of forces are added to d
                for(int k=0; k<dim; k++)
                    for(int l=0; l<nb; l++)
                        d[j+l] += w[j+l][k] * force[k];


                //3. the specific resolution of the constraint(s) is called
                res[j]->resolution(j, w, d, force, dfree);

                //4. the error is measured (displacement due to the new resolution (i.e. due to the new force))
                SReal contraintError = 0.0;
                if(nb > 1)
                {
                    for(int l=0; l<nb; l++)
                    {
                        SReal lineError = 0.0;
                        for (int m=0; m<nb; m++)
                        {
                            SReal dofError = w[j+l][j+m] * (force[j+m] - errF[m]);
                            lineError += dofError * dofError;
                        }
                        lineError = sqrt(lineError);
                        if(lineError > tolerance)
                            constraintsAreVerified = false;

                        contraintError += lineError;
                    }
                }
                else
                {
                    contraintError = fabs(w[j][j] * (force[j] - errF[0]));
                    if(contraintError > tolerance)
                        constraintsAreVerified = false;
                }

                if(res[j]->getTolerance())
                {
                    if(contraintError > res[j]->getTolerance())
                        constraintsAreVerified = false;
                    contraintError *= tolerance / res[j]->getTolerance();
                }

                error += contraintError;
                tabErrors[j] = contraintError;

                j += nb;
            }
            else
            {
                std::fill_n(&force[j], nb, 0);
                msg_info_when(iter==0) << "constraint %d has a compliance equal to zero on the diagonal" ;
                j += nb;
            }
        }


        /// display a graph with the force of each constraint dimension at each iteration
        std::map < std::string, sofa::type::vector<SReal> >* graphs = d_graphForces.beginEdit();
        for(int j=0; j<dim; j++)
        {
            std::ostringstream oss;
            oss << "f" << j;

            sofa::type::vector<SReal>& graph_force = (*graphs)[oss.str()];
            graph_force.push_back(force[j]);
        }
        d_graphForces.endEdit();

        graph_residuals.push_back(error);

        if(sor != 1.0)
        {
            for(int j=0; j<dim; j++)
                force[j] = sor * force[j] + (1-sor) * tempForces[j];
        }

        if(allVerified)
        {
            if(constraintsAreVerified)
            {
                convergence = true;
                break;
            }
        }
        else if(error < tolerance && iter>0) // do not stop at the first iteration (that is used for initial guess computation)
        {
            convergence = true;
            break;
        }
    }

    if (EMIT_EXTRA_DEBUG_MESSAGE)
    {
        if (!convergence)
        {
            msg_error() << "No convergence in gaussSeidelConstraint : error = " << error;
        }
        else if (d_displayTime.getValue())
        {
            msg_info() << "Convergence after " << iter+1 << " iterations.";
        }
    }

    sofa::helper::AdvancedTimer::valSet("GS iterations", iter+1);

    for(int i=0; i<dim; )
    {
        res[i]->store(i, force, convergence);
        int t = res[i]->getNbLines();
        i += t;
    }

    if(d_schemeCorrection.getValue())
    {
        ///////// scheme correction : step 3 => the corrective motion is only based on the diff of the force value: compute this diff
        for(int j=0; j<dim; j++)
        {
            df[j] += force[j];
        }
    }


    ////////// DISPLAY A GRAPH WITH THE CONVERGENCE PERF ON THE GUI :
    d_graphErrors.endEdit();

    sofa::type::vector<SReal>& graph_constraints = (*d_graphConstraints.beginEdit())["Constraints"];
    graph_constraints.clear();

    for(int j=0; j<dim; )
    {
        nb = res[j]->getNbLines();

        if(tabErrors[j])
            graph_constraints.push_back(tabErrors[j]);
        else if(res[j]->getTolerance())
            graph_constraints.push_back(res[j]->getTolerance());
        else
            graph_constraints.push_back(tolerance);

        j += nb;
    }
    d_graphConstraints.endEdit();
}




void ConstraintAnimationLoop::debugWithContact(int numConstraints)
{
    const SReal mu=0.8;
    ConstraintProblem& CP = (d_doubleBuffer.getValue() && bufCP1) ? CP2 : CP1;
    helper::nlcp_gaussseidel(numConstraints, CP.getDfree()->ptr(), CP.getW()->lptr(), CP.getF()->ptr(), mu, d_tol.getValue(), d_maxIt.getValue(), false, EMIT_EXTRA_DEBUG_MESSAGE);
    CP.getF()->clear();
    CP.getF()->resize(numConstraints);

}

ConstraintProblem* ConstraintAnimationLoop::getCP()
{
    if (d_doubleBuffer.getValue() && bufCP1)
        return &CP2;
    else
        return &CP1;
}

void registerConstraintAnimationLoop(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Constraint animation loop manager")
        .add< ConstraintAnimationLoop >());
}

} //namespace sofa::component::animationloop
