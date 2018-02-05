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
#include <SofaConstraint/ConstraintAnimationLoop.h>
#include <sofa/core/visual/VisualParams.h>

#include <SofaConstraint/ConstraintSolverImpl.h>

#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/SolveVisitor.h>
#include <sofa/simulation/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/UpdateMappingEndEvent.h>
#include <sofa/simulation/UpdateBoundingBoxVisitor.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/helper/LCPcalc.h>

#include <sofa/core/ConstraintParams.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/helper/system/thread/CTime.h>
#include <sofa/helper/AdvancedTimer.h>

#include <math.h>

#include <map>
#include <string>
#include <sstream>

/// Change that to true if you want to print extra message on this component.
/// You can eventually link that to an object attribute.
#define EMIT_EXTRA_DEBUG_MESSAGE false

using namespace sofa::defaulttype;
using namespace sofa::component::linearsolver;
using namespace sofa::helper::system::thread;

namespace sofa
{

namespace component
{

namespace animationloop
{

using namespace sofa::defaulttype;
using namespace helper::system::thread;
using namespace core::behavior;
using namespace sofa::simulation;


ConstraintProblem::ConstraintProblem(bool printLog)
    : m_printLog(printLog)
{
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
        if (_constraintsResolutions[i] != NULL)
        {
            delete _constraintsResolutions[i];
            _constraintsResolutions[i] = NULL;
        }
    }
    _constraintsResolutions.clear(); // _constraintsResolutions.clear();
    delete(_timer);
}

void ConstraintProblem::clear(int dim, const double &tol)
{
    // if not null delete the old constraintProblem
    for(int i=0; i<_dim; i++)
    {
        if (_constraintsResolutions[i] != NULL)
        {
            delete _constraintsResolutions[i];
            _constraintsResolutions[i] = NULL;
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


void ConstraintProblem::gaussSeidelConstraintTimed(double &timeout, int numItMax)
{
    int i, j, k, l, nb;

    double errF[6] = {0,0,0,0,0,0};
    double error=0.0;



    double t0 = (double)_timer->getTime() ;
    double timeScale = 1.0 / (double)CTime::getTicksPerSec();

    for(i=0; i<numItMax; i++)
    {
        error=0.0;
        for(j=0; j<_dim; ) // increment of j realized at the end of the loop
        {
            nb = _constraintsResolutions[j]->nbLines;

            //2. for each line we compute the actual value of d
            //   (a)d is set to dfree
            for(l=0; l<nb; l++)
            {
                errF[l] = _force[j+l];
                _d[j+l] = _dFree[j+l];
            }

            //   (b) contribution of forces are added to d
            for(k=0; k<_dim; k++)
                for(l=0; l<nb; l++)
                    _d[j+l] += _W[j+l][k] * _force[k];



            //3. the specific resolution of the constraint(s) is called
            //double** w = this->_W.ptr();
            _constraintsResolutions[j]->resolution(j, this->getW()->lptr(), this->getD()->ptr(), this->getF()->ptr(), _dFree.ptr());

            //4. the error is measured (displacement due to the new resolution (i.e. due to the new force))
            if(nb > 1)
            {
                double terr = 0.0;
                for(l=0; l<nb; l++)
                {
                    double terr2 = 0;
                    for (int m=0; m<nb; m++)
                    {
                        terr2 += _W[j+l][j+m] * (_force[j+m] - errF[m]);
                    }
                    terr += terr2 * terr2;
                }
                error += sqrt(terr);
            }
            else
                error += fabs(_W[j][j] * (_force[j] - errF[0]));

            j += nb;
        }

        /////////////////// GAUSS SEIDEL IS TIMED !!! /////////
        double t1 = (double)_timer->getTime();
        double dt = (t1 - t0)*timeScale;
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


ConstraintAnimationLoop::ConstraintAnimationLoop(simulation::Node* gnode)
    : Inherit(gnode)
    , displayTime(initData(&displayTime, false, "displayTime","Display time for each important step of ConstraintAnimationLoop."))
    , _tol( initData(&_tol, 0.00001, "tolerance", "Tolerance of the Gauss-Seidel"))
    , _maxIt( initData(&_maxIt, 1000, "maxIterations", "Maximum number of iterations of the Gauss-Seidel"))
    , doCollisionsFirst(initData(&doCollisionsFirst, false, "doCollisionsFirst","Compute the collisions first (to support penality-based contacts)"))
    , doubleBuffer( initData(&doubleBuffer, false, "doubleBuffer","Buffer the constraint problem in a double buffer to be accessible with an other thread"))
    , scaleTolerance( initData(&scaleTolerance, true, "scaleTolerance","Scale the error tolerance with the number of constraints"))
    , _allVerified( initData(&_allVerified, false, "allVerified","All contraints must be verified (each constraint's error < tolerance)"))
    , _sor( initData(&_sor, 1.0, "sor","Successive Over Relaxation parameter (0-2)"))
    , schemeCorrection( initData(&schemeCorrection, false, "schemeCorrection","Apply new scheme where compliance is progressively corrected"))
    , _realTimeCompensation( initData(&_realTimeCompensation, false, "realTimeCompensation","If the total computational time T < dt, sleep(dt-T)"))
    , _graphErrors( initData(&_graphErrors,"graphErrors","Sum of the constraints' errors at each iteration"))
    , _graphConstraints( initData(&_graphConstraints,"graphConstraints","Graph of each constraint's error at the end of the resolution"))
    , _graphForces( initData(&_graphForces,"graphForces","Graph of each constraint's force at each step of the resolution"))
{
    bufCP1 = false;

    _graphErrors.setWidget("graph");
    //	_graphErrors.setReadOnly(true);
    _graphErrors.setGroup("Graph");

    _graphConstraints.setWidget("graph");
    //	_graphConstraints.setReadOnly(true);
    _graphConstraints.setGroup("Graph");

    _graphForces.setWidget("graph");
    //	_graphForces.setReadOnly(true);
    _graphForces.setGroup("Graph2");

    CP1.clear(0,_tol.getValue());
    CP2.clear(0,_tol.getValue());

    timer = 0;

    msg_deprecated("ConstraintAnimationLoop") << "WARNING : ConstraintAnimationLoop is deprecated. Please use the combination of FreeMotionAnimationLoop and GenericConstraintSolver." ;
}

ConstraintAnimationLoop::~ConstraintAnimationLoop()
{
    if (timer != 0)
    {
        delete timer;
        timer = 0;
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
            <<"computeCollision is called"<<sendl;

    ////////////////// COLLISION DETECTION///////////////////////////////////////////////////////////////////////////////////////////
    sofa::helper::AdvancedTimer::stepBegin("Collision");
    computeCollision(params);
    sofa::helper::AdvancedTimer::stepEnd  ("Collision");
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if ( displayTime.getValue() )
    {
        sout<<" computeCollision                 " << ( (double) timer->getTime() - time)*timeScale <<" ms" <<sendl;
        time = (SReal) timer->getTime();
    }

}


void ConstraintAnimationLoop::freeMotion(const core::ExecParams* params, simulation::Node *context, SReal &dt)
{
    dmsg_info_when(EMIT_EXTRA_DEBUG_MESSAGE)
            <<"Free Motion is called" ;

    ///////////////////////////////////////////// FREE MOTION /////////////////////////////////////////////////////////////
    sofa::helper::AdvancedTimer::stepBegin("Free Motion");
    simulation::MechanicalBeginIntegrationVisitor(params, dt).execute(context);

    ////////////////// (optional) PREDICTIVE CONSTRAINT FORCES ///////////////////////////////////////////////////////////////////////////////////////////
    /// When scheme Correction is used, the constraint forces computed at the previous time-step
    /// are applied during the first motion, so which is no more a "free" motion but a "predictive" motion
    ///////////
    if(schemeCorrection.getValue())
    {
        sofa::core::ConstraintParams cparams(*params);
        sofa::core::MultiVecDerivId f =  core::VecDerivId::externalForce();

        for (unsigned int i=0; i<constraintCorrections.size(); i++ )
        {
            core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
            cc->applyPredictiveConstraintForce(&cparams, f, getCP()->getF());
        }
    }

    simulation::SolveVisitor(params, dt, true).execute(context);

    {
        sofa::core::MechanicalParams mparams(*params);
        sofa::core::MultiVecCoordId xfree = sofa::core::VecCoordId::freePosition();
        mparams.x() = xfree;
        simulation::MechanicalProjectPositionVisitor(&mparams, 0, xfree ).execute(context);
        simulation::MechanicalPropagateOnlyPositionVisitor(&mparams, 0, xfree, true ).execute(context);
    }
    sofa::helper::AdvancedTimer::stepEnd  ("Free Motion");

    //////// TODO : propagate velocity !!

    ////////propagate acceleration ? //////

    //this is done to set dx to zero in subgraph
    core::MultiVecDerivId dx_id = core::VecDerivId::dx();
    simulation::MechanicalVOpVisitor(params, dx_id, core::ConstVecId::null(), core::ConstVecId::null(), 1.0 ).setMapped(true).execute(context);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if ( displayTime.getValue() )
    {
        sout << ">>>>> Begin display ConstraintAnimationLoop time" << sendl;
        sout<<" Free Motion                           " << ( (double) timer->getTime() - time)*timeScale <<" ms" <<sendl;
        time = (double) timer->getTime();
    }
}

void ConstraintAnimationLoop::setConstraintEquations(const core::ExecParams* params, simulation::Node *context)
{
    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetContactForce();
    }

    //////////////////////////////////////CONSTRAINTS RESOLUTION//////////////////////////////////////////////////////////////////////
    msg_info_when(EMIT_EXTRA_DEBUG_MESSAGE) <<"constraints Matrix construction is called" ;

    sofa::helper::AdvancedTimer::stepBegin("Constraints definition");


    if(!schemeCorrection.getValue())
    {
        /// calling resetConstraint & setConstraint & accumulateConstraint visitors
        /// and resize the constraint problem that will be solved
        unsigned int numConstraints = 0;
        writeAndAccumulateAndCountConstraintDirections(params, context, numConstraints);
    }


    core::MechanicalParams mparams = core::MechanicalParams(*params);
    simulation::MechanicalProjectJacobianMatrixVisitor(&mparams).execute(context);

    /// calling GetConstraintViolationVisitor: each constraint provides its present violation
    /// for a given state (by default: free_position TODO: add VecId to make this method more generic)
    getIndividualConstraintViolations(params, context);

    if(!schemeCorrection.getValue())
    {
        /// calling getConstraintResolution: each constraint provides a method that is used to solve it during GS iterations
        getIndividualConstraintSolvingProcess(params, context);
    }

    sofa::helper::AdvancedTimer::stepEnd  ("Constraints definition");

    /// calling getCompliance projected in the contact space => getDelassusOperator(_W) = H*C*Ht
    computeComplianceInConstraintSpace();

    if ( displayTime.getValue() )
    {
        sout<<" Build problem in the constraint space " << ( (double) timer->getTime() - time)*timeScale<<" ms" <<sendl;
        time = (double) timer->getTime();
    }
}

void ConstraintAnimationLoop::writeAndAccumulateAndCountConstraintDirections(const core::ExecParams* params, simulation::Node *context, unsigned int &numConstraints)
{
    // calling resetConstraint on LMConstraints and MechanicalStates
    simulation::MechanicalResetConstraintVisitor(params).execute(context);

    core::ConstraintParams cparams = core::ConstraintParams(*params);
    cparams.setX(core::ConstVecCoordId::freePosition());
    cparams.setV(core::ConstVecDerivId::freeVelocity());

    // calling applyConstraint on each constraint

    MechanicalSetConstraint(&cparams, core::MatrixDerivId::constraintJacobian(), numConstraints).execute(context);

    sofa::helper::AdvancedTimer::valSet("numConstraints", numConstraints);

    // calling accumulateConstraint on the mappings
    MechanicalAccumulateConstraint2(&cparams, core::MatrixDerivId::constraintJacobian()).execute(context);

    getCP()->clear(numConstraints,this->_tol.getValue());
}

void ConstraintAnimationLoop::getIndividualConstraintViolations(const core::ExecParams* params, simulation::Node *context)
{
    core::ConstraintParams cparams = core::ConstraintParams(*params);
    cparams.setX(core::ConstVecCoordId::freePosition());
    cparams.setV(core::ConstVecDerivId::freeVelocity());

    constraintset::MechanicalGetConstraintViolationVisitor(&cparams, getCP()->getDfree()).execute(context);
}

void ConstraintAnimationLoop::getIndividualConstraintSolvingProcess(const core::ExecParams* params, simulation::Node *context)
{
    /// calling getConstraintResolution: each constraint provides a method that is used to solve it during GS iterations
    core::ConstraintParams cparams = core::ConstraintParams(*params);
    cparams.setX(core::ConstVecCoordId::freePosition());
    cparams.setV(core::ConstVecDerivId::freeVelocity());

    MechanicalGetConstraintResolutionVisitor(&cparams, getCP()->getConstraintResolutions(), 0).execute(context);
}

void ConstraintAnimationLoop::computeComplianceInConstraintSpace()
{
    /// calling getCompliance => getDelassusOperator(_W) = H*C*Ht
    dmsg_info_when(EMIT_EXTRA_DEBUG_MESSAGE) << "   4. get Compliance " ;

    sofa::helper::AdvancedTimer::stepBegin("Get Compliance");
    for (unsigned int i=0; i<constraintCorrections.size(); i++ )
    {
        core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->addComplianceInConstraintSpace(core::ConstraintParams::defaultInstance(), getCP()->getW());
    }

    sofa::helper::AdvancedTimer::stepEnd  ("Get Compliance");

}

void ConstraintAnimationLoop::correctiveMotion(const core::ExecParams* params, simulation::Node *context)
{
    dmsg_info_when(EMIT_EXTRA_DEBUG_MESSAGE)
            <<"constraintCorrections motion is called" ;

    sofa::helper::AdvancedTimer::stepBegin("Corrective Motion");

    if(schemeCorrection.getValue())
    {
        // IF SCHEME CORRECTIVE=> correct the motion using dF
        for (unsigned int i=0; i<constraintCorrections.size(); i++)
        {
            core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
            cc->applyContactForce(getCP()->getdF());
        }
    }
    else
    {
        // ELSE => only correct the motion using F
        for (unsigned int i=0; i<constraintCorrections.size(); i++)
        {
            core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
            cc->applyContactForce(getCP()->getF());
        }
    }

    simulation::common::MechanicalOperations mop(params, this->getContext());

    mop.propagateV(core::VecDerivId::velocity());

    mop.propagateDx(core::VecDerivId::dx(), true);

    // "mapped" x = xfree + dx
    simulation::MechanicalVOpVisitor(params, core::VecCoordId::position(), core::ConstVecCoordId::freePosition(), core::ConstVecDerivId::dx(), 1.0 ).setOnlyMapped(true).execute(context);

    if(!schemeCorrection.getValue())
    {
        for (unsigned int i=0; i<constraintCorrections.size(); i++)
        {
            core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
            cc->resetContactForce();
        }
    }

    sofa::helper::AdvancedTimer::stepEnd ("Corrective Motion");
}

void ConstraintAnimationLoop::step ( const core::ExecParams* params, SReal dt )
{

    static SReal simulationTime=0.0;

    simulationTime+=dt;
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Step");
#endif

    {
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        this->gnode->execute ( act );
    }


    SReal startTime = this->gnode->getTime();

    BehaviorUpdatePositionVisitor beh(params , this->gnode->getDt());
    this->gnode->execute ( beh );


    if (simulationTime>0.1)
        activateSubGraph.setValue(true);
    else
        activateSubGraph.setValue(false);

    time = 0.0;
    SReal totaltime = 0.0;
    timeScale = 1.0 / (SReal)CTime::getTicksPerSec() * 1000;
    if ( displayTime.getValue() )
    {
        if (timer == 0)
            timer = new CTime();

        time = (SReal) timer->getTime();
        totaltime = time;
        sout<<sendl;
    }
    if (doubleBuffer.getValue())
    {
        // SWAP BUFFER:
        bufCP1 = !bufCP1;
    }

    ConstraintProblem& CP = (doubleBuffer.getValue() && bufCP1) ? CP2 : CP1;

#if !defined(WIN32) && !defined(_XBOX)
    if (_realTimeCompensation.getValue())
    {
        if (timer == 0)
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
                usleep(toSleep);
            else
                serr << "Cannot achieve frequency for dt = " << dt << sendl;
            compTime = (SReal)timer->getTime();
        }
    }
#endif

    dmsg_info() << " step is called" ;

    // This solver will work in freePosition and freeVelocity vectors.
    // We need to initialize them if it's not already done.
    simulation::MechanicalVInitVisitor<core::V_COORD>(params, core::VecCoordId::freePosition(), core::ConstVecCoordId::position(), true).execute(this->gnode);
    simulation::MechanicalVInitVisitor<core::V_DERIV>(params, core::VecDerivId::freeVelocity(), core::ConstVecDerivId::velocity()).execute(this->gnode);

    if (doCollisionsFirst.getValue())
    {
        /// COLLISION
        launchCollisionDetection(params);
    }

    // Update the BehaviorModels => to be removed ?
    // Required to allow the RayPickInteractor interaction
    sofa::helper::AdvancedTimer::stepBegin("BehaviorUpdate");
    simulation::BehaviorUpdatePositionVisitor(params, dt).execute(this->gnode);
    sofa::helper::AdvancedTimer::stepEnd  ("BehaviorUpdate");


    if(schemeCorrection.getValue())
    {
        // Compute the predictive force:
        numConstraints = 0;

        //1. Find the new constraint direction
        writeAndAccumulateAndCountConstraintDirections(params, this->gnode, numConstraints);

        //2. Get the constraint solving process:
        getIndividualConstraintSolvingProcess(params, this->gnode);

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
    freeMotion(params, this->gnode, dt);



    if (!doCollisionsFirst.getValue())
    {
        /// COLLISION
        launchCollisionDetection(params);
    }

    //////////////// BEFORE APPLYING CONSTRAINT  : propagate position through mapping
    core::MechanicalParams mparams(*params);
    simulation::MechanicalProjectPositionVisitor(&mparams, 0, core::VecCoordId::position()).execute(this->gnode);
    simulation::MechanicalPropagateOnlyPositionVisitor(&mparams, 0, core::VecCoordId::position(), true).execute(this->gnode);


    /// CONSTRAINT SPACE & COMPLIANCE COMPUTATION
    setConstraintEquations(params, this->gnode);

    if (EMIT_EXTRA_DEBUG_MESSAGE)
    {
        msg_info() << "getF() after setConstraintEquations:" ;
        helper::resultToString(std::cout, CP.getF()->ptr(),CP.getSize());
    }

    sofa::helper::AdvancedTimer::stepBegin("GaussSeidel");

    if (EMIT_EXTRA_DEBUG_MESSAGE)
        msg_info() << "Gauss-Seidel solver is called on problem of size " << CP.getSize() ;

    if(schemeCorrection.getValue())
        (*CP.getF())*=0.0;

    gaussSeidelConstraint(CP.getSize(), CP.getDfree()->ptr(), CP.getW()->lptr(), CP.getF()->ptr(), CP.getD()->ptr(), CP.getConstraintResolutions(), CP.getdF()->ptr());

    sofa::helper::AdvancedTimer::stepEnd  ("GaussSeidel");

    if (EMIT_EXTRA_DEBUG_MESSAGE)
        helper::afficheLCP(CP.getDfree()->ptr(), CP.getW()->lptr(), CP.getF()->ptr(),  CP.getSize());

    if ( displayTime.getValue() )
    {
        msg_info() << " Solve with GaussSeidel                " << ( (SReal) timer->getTime() - time)*timeScale<<" ms" ;
        time = (SReal) timer->getTime();
    }

    /// CORRECTIVE MOTION
    correctiveMotion(params, this->gnode);

    if ( displayTime.getValue() )
    {
        msg_info() << " ContactCorrections                    " << ( (SReal) timer->getTime() - time)*timeScale <<" ms" << msgendl
                   << "  = Total                              " << ( (SReal) timer->getTime() - totaltime)*timeScale <<" ms" << msgendl
                   << " With : " << CP.getSize() << " constraints" << msgendl
                   << "<<<<< End display ConstraintAnimationLoop time." ;
    }

    simulation::MechanicalEndIntegrationVisitor endVisitor(params, dt);
    this->gnode->execute(&endVisitor);
    this->gnode->setTime ( startTime + dt );
    this->gnode->execute<UpdateSimulationContextVisitor>(params);  // propagate time

    {
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        this->gnode->execute ( act );
    }

    sofa::helper::AdvancedTimer::stepBegin("UpdateMapping");

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


}

void ConstraintAnimationLoop::computePredictiveForce(int dim, double* force, std::vector<core::behavior::ConstraintResolution*>& res)
{
    for(int i=0; i<dim; )
    {
        res[i]->initForce(i, force);
        i += res[i]->nbLines;
    }
}

void ConstraintAnimationLoop::gaussSeidelConstraint(int dim, double* dfree, double** w, double* force,
        double* d, std::vector<ConstraintResolution*>& res, double* df=NULL)
{
    if(!dim)
        return;

    int i, j, k, l, nb;
    double errF[6] = {0,0,0,0,0,0};
    double error=0.0;

    double tolerance = _tol.getValue();
    int numItMax = _maxIt.getValue();
    bool convergence = false;
    double sor = _sor.getValue();
    bool allVerified = _allVerified.getValue();
    sofa::helper::vector<double> tempForces;
    if(sor != 1.0) tempForces.resize(dim);

    if(scaleTolerance.getValue() && !allVerified)
        tolerance *= dim;

    for(i=0; i<dim; )
    {
        res[i]->init(i, w, force);
        i += res[i]->nbLines;
    }

    std::map < std::string, sofa::helper::vector<double> >* graphs = _graphForces.beginEdit();
    graphs->clear();
    _graphForces.endEdit();

    if(schemeCorrection.getValue())
    {
        std::cout<<"shemeCorrection => LCP before step 1"<<std::endl;
        helper::afficheLCP(dfree, w, force,  dim);
        ///////// scheme correction : step 1 => modification of dfree
        for(j=0; j<dim; j++)
        {
            for(k=0; k<dim; k++)
                dfree[j] -= w[j][k] * force[k];
        }

        ///////// scheme correction : step 2 => storage of force value
        for(j=0; j<dim; j++)
            df[j] = -force[j];
    }

    sofa::helper::vector<double>& graph_residuals = (*_graphErrors.beginEdit())["Error"];
    graph_residuals.clear();

    sofa::helper::vector<double> tabErrors;
    tabErrors.resize(dim);

    for(i=0; i<numItMax; i++)
    {
        bool constraintsAreVerified = true;
        if(sor != 1.0)
        {
            for(j=0; j<dim; j++)
                tempForces[j] = force[j];
        }

        error=0.0;

        for(j=0; j<dim; ) // increment of j realized at the end of the loop
        {
            //1. nbLines provide the dimension of the constraint  (max=6)
            nb = res[j]->nbLines;

            bool check = true;
            for (int b=0; b<nb; b++)
            {
                if (w[j+b][j+b]==0.0) check = false;
            }

            if (check)
            {
                //2. for each line we compute the actual value of d
                //   (a)d is set to dfree
                for(l=0; l<nb; l++)
                {
                    errF[l] = force[j+l];
                    d[j+l] = dfree[j+l];
                }
                //   (b) contribution of forces are added to d
                for(k=0; k<dim; k++)
                    for(l=0; l<nb; l++)
                        d[j+l] += w[j+l][k] * force[k];


                //3. the specific resolution of the constraint(s) is called
                res[j]->resolution(j, w, d, force, dfree);

                //4. the error is measured (displacement due to the new resolution (i.e. due to the new force))
                double contraintError = 0.0;
                if(nb > 1)
                {
                    for(l=0; l<nb; l++)
                    {
                        double lineError = 0.0;
                        for (int m=0; m<nb; m++)
                        {
                            double dofError = w[j+l][j+m] * (force[j+m] - errF[m]);
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

                if(res[j]->tolerance)
                {
                    if(contraintError > res[j]->tolerance)
                        constraintsAreVerified = false;
                    contraintError *= tolerance / res[j]->tolerance;
                }

                error += contraintError;
                tabErrors[j] = contraintError;

                j += nb;
            }
            else
            {
                for (int b=0; b<nb; b++) force[j+b] = 0;
                msg_info_when(i==0) << "constraint %d has a compliance equal to zero on the diagonal" ;
                j += nb;
            }
        }


        /// display a graph with the force of each constraint dimension at each iteration
        std::map < std::string, sofa::helper::vector<double> >* graphs = _graphForces.beginEdit();
        for(j=0; j<dim; j++)
        {
            std::ostringstream oss;
            oss << "f" << j;

            sofa::helper::vector<double>& graph_force = (*graphs)[oss.str()];
            graph_force.push_back(force[j]);
        }
        _graphForces.endEdit();

        graph_residuals.push_back(error);

        if(sor != 1.0)
        {
            for(j=0; j<dim; j++)
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
        else if(error < tolerance && i>0) // do not stop at the first iteration (that is used for initial guess computation)
        {
            convergence = true;
            break;
        }
    }

    if (EMIT_EXTRA_DEBUG_MESSAGE)
    {
        if (!convergence)
        {
            serr << "No convergence in gaussSeidelConstraint : error = " << error << sendl;
        }
        else if (displayTime.getValue())
        {
            sout << "Convergence after " << i+1 << " iterations." << sendl;
        }
    }

    sofa::helper::AdvancedTimer::valSet("GS iterations", i+1);

    for(i=0; i<dim; )
    {
        res[i]->store(i, force, convergence);
        int t = res[i]->nbLines;
        i += t;
    }

    if(schemeCorrection.getValue())
    {
        ///////// scheme correction : step 3 => the corrective motion is only based on the diff of the force value: compute this diff
        for(j=0; j<dim; j++)
        {
            df[j] += force[j];
        }
    }


    ////////// DISPLAY A GRAPH WITH THE CONVERGENCE PERF ON THE GUI :
    _graphErrors.endEdit();

    sofa::helper::vector<double>& graph_constraints = (*_graphConstraints.beginEdit())["Constraints"];
    graph_constraints.clear();

    for(j=0; j<dim; )
    {
        nb = res[j]->nbLines;

        if(tabErrors[j])
            graph_constraints.push_back(tabErrors[j]);
        else if(res[j]->tolerance)
            graph_constraints.push_back(res[j]->tolerance);
        else
            graph_constraints.push_back(tolerance);

        j += nb;
    }
    _graphConstraints.endEdit();
}




void ConstraintAnimationLoop::debugWithContact(int numConstraints)
{

    double mu=0.8;
    ConstraintProblem& CP = (doubleBuffer.getValue() && bufCP1) ? CP2 : CP1;
    helper::nlcp_gaussseidel(numConstraints, CP.getDfree()->ptr(), CP.getW()->lptr(), CP.getF()->ptr(), mu, _tol.getValue(), _maxIt.getValue(), false, EMIT_EXTRA_DEBUG_MESSAGE);
    CP.getF()->clear();
    CP.getF()->resize(numConstraints);

}


SOFA_DECL_CLASS ( ConstraintAnimationLoop )

int ConstraintAnimationLoopClass = core::RegisterObject ( "Constraint animation loop manager" )
        .add< ConstraintAnimationLoop >()
        .addAlias("MasterConstraintSolver");

} // namespace animationloop

} // namespace component

} // namespace sofa
