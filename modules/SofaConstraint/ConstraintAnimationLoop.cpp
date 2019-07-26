/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <cmath>

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

sofa::simulation::Visitor::Result MechanicalGetConstraintResolutionVisitor::fwdConstraintSet(simulation::Node* node, core::behavior::BaseConstraintSet* cSet)
{
    if (core::behavior::BaseConstraint *c=cSet->toBaseConstraint())
    {
        ctime_t t0 = begin(node, c);
        c->getConstraintResolution(_cparams, _res, _offset);
        end(node, c, t0);
    }
    return RESULT_CONTINUE;
}

bool MechanicalGetConstraintResolutionVisitor::stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
{
    return false; // !map->isMechanical();
}

sofa::simulation::Visitor::Result MechanicalSetConstraint::fwdConstraintSet(simulation::Node* node, core::behavior::BaseConstraintSet* c)
{
    ctime_t t0 = begin(node, c);

    c->setConstraintId(contactId);
    c->buildConstraintMatrix(cparams, res, contactId);

    end(node, c, t0);
    return RESULT_CONTINUE;
}

const char* MechanicalSetConstraint::getClassName() const
{
    return "MechanicalSetConstraint";
}

bool MechanicalSetConstraint::isThreadSafe() const
{
    return false;
}

bool MechanicalSetConstraint::stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
{
    return false; // !map->isMechanical();
}

void MechanicalAccumulateConstraint2::bwdMechanicalMapping(simulation::Node* node, core::BaseMapping* map)
{
    ctime_t t0 = begin(node, map);
    map->applyJT(cparams, res, res);
    end(node, map, t0);
}

const char* MechanicalAccumulateConstraint2::getClassName() const
{
    return "MechanicalAccumulateConstraint2";
}

bool MechanicalAccumulateConstraint2::isThreadSafe() const
{
    return false;
}

bool MechanicalAccumulateConstraint2::stopAtMechanicalMapping(simulation::Node* /*node*/, core::BaseMapping* /*map*/)
{
    return false; // !map->isMechanical();
}

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
    double error=0.0;

    double t0 = (double)_timer->getTime() ;
    double timeScale = 1.0 / (double)CTime::getTicksPerSec();

    for(int i=0; i<numItMax; i++)
    {
        error=0.0;
        for(int j=0; j<_dim; ) // increment of j realized at the end of the loop
        {
            int nb = _constraintsResolutions[j]->getNbLines();

            //2. for each line we compute the actual value of d
            //   (a)d is set to dfree
            std::vector<double> errF(&_force[j], &_force[j+nb]);
            std::copy_n(_dFree.begin() + j, nb, _d.begin() + j);

            //   (b) contribution of forces are added to d
            for(int k=0; k<_dim; k++)
                for(int l=0; l<nb; l++)
                    _d[j+l] += _W[j+l][k] * _force[k];



            //3. the specific resolution of the constraint(s) is called
            //double** w = this->_W.ptr();
            _constraintsResolutions[j]->resolution(j, this->getW()->lptr(), this->getD()->ptr(), this->getF()->ptr(), _dFree.ptr());

            //4. the error is measured (displacement due to the new resolution (i.e. due to the new force))
            if(nb > 1)
            {
                double terr = 0.0;
                for(int l=0; l<nb; l++)
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
    , d_displayTime(initData(&d_displayTime, false, "displayTime","Display time for each important step of ConstraintAnimationLoop."))
    , d_tol( initData(&d_tol, 0.00001, "tolerance", "Tolerance of the Gauss-Seidel"))
    , d_maxIt( initData(&d_maxIt, 1000, "maxIterations", "Maximum number of iterations of the Gauss-Seidel"))
    , d_doCollisionsFirst(initData(&d_doCollisionsFirst, false, "doCollisionsFirst","Compute the collisions first (to support penality-based contacts)"))
    , d_doubleBuffer( initData(&d_doubleBuffer, false, "doubleBuffer","Buffer the constraint problem in a double buffer to be accessible with an other thread"))
    , d_scaleTolerance( initData(&d_scaleTolerance, true, "scaleTolerance","Scale the error tolerance with the number of constraints"))
    , d_allVerified( initData(&d_allVerified, false, "allVerified","All contraints must be verified (each constraint's error < tolerance)"))
    , d_sor( initData(&d_sor, 1.0, "sor","Successive Over Relaxation parameter (0-2)"))
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
            <<"computeCollision is called"<<sendl;

    ////////////////// COLLISION DETECTION///////////////////////////////////////////////////////////////////////////////////////////
    sofa::helper::AdvancedTimer::stepBegin("Collision");
    computeCollision(params);
    sofa::helper::AdvancedTimer::stepEnd  ("Collision");
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if ( d_displayTime.getValue() )
    {
        msg_info() <<" computeCollision                 " << ( (double) timer->getTime() - time)*timeScale <<" ms";
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
    if(d_schemeCorrection.getValue())
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

    if ( d_displayTime.getValue() )
    {
        msg_info() << ">>>>> Begin display ConstraintAnimationLoop time";
        msg_info() <<" Free Motion                           " << ( (double) timer->getTime() - time)*timeScale <<" ms";
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


    if(!d_schemeCorrection.getValue())
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

    if(!d_schemeCorrection.getValue())
    {
        /// calling getConstraintResolution: each constraint provides a method that is used to solve it during GS iterations
        getIndividualConstraintSolvingProcess(params, context);
    }

    sofa::helper::AdvancedTimer::stepEnd  ("Constraints definition");

    /// calling getCompliance projected in the contact space => getDelassusOperator(_W) = H*C*Ht
    computeComplianceInConstraintSpace();

    if ( d_displayTime.getValue() )
    {
        msg_info()<<" Build problem in the constraint space " << ( (double) timer->getTime() - time)*timeScale<<" ms";
        time = (double) timer->getTime();
    }
}

void ConstraintAnimationLoop::writeAndAccumulateAndCountConstraintDirections(const core::ExecParams* params, simulation::Node *context, unsigned int &numConstraints)
{
    core::ConstraintParams cparams = core::ConstraintParams(*params);
    cparams.setX(core::ConstVecCoordId::freePosition());
    cparams.setV(core::ConstVecDerivId::freeVelocity());

    // calling resetConstraint on LMConstraints and MechanicalStates
    simulation::MechanicalResetConstraintVisitor(&cparams).execute(context);

    // calling applyConstraint on each constraint
    MechanicalSetConstraint(&cparams, core::MatrixDerivId::constraintJacobian(), numConstraints).execute(context);

    sofa::helper::AdvancedTimer::valSet("numConstraints", numConstraints);

    // calling accumulateConstraint on the mappings
    MechanicalAccumulateConstraint2(&cparams, core::MatrixDerivId::constraintJacobian()).execute(context);

    getCP()->clear(numConstraints,this->d_tol.getValue());
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

    if(d_schemeCorrection.getValue())
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

    if(!d_schemeCorrection.getValue())
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
                msg_error() << "Cannot achieve frequency for dt = " << dt ;
            compTime = (SReal)timer->getTime();
        }
    }
#endif

    dmsg_info() << " step is called" ;

    // This solver will work in freePosition and freeVelocity vectors.
    // We need to initialize them if it's not already done.
    simulation::MechanicalVInitVisitor<core::V_COORD>(params, core::VecCoordId::freePosition(), core::ConstVecCoordId::position(), true).execute(this->gnode);
    simulation::MechanicalVInitVisitor<core::V_DERIV>(params, core::VecDerivId::freeVelocity(), core::ConstVecDerivId::velocity()).execute(this->gnode);

    if (d_doCollisionsFirst.getValue())
    {
        /// COLLISION
        launchCollisionDetection(params);
    }

    // Update the BehaviorModels => to be removed ?
    // Required to allow the RayPickInteractor interaction
    sofa::helper::AdvancedTimer::stepBegin("BehaviorUpdate");
    simulation::BehaviorUpdatePositionVisitor(params, dt).execute(this->gnode);
    sofa::helper::AdvancedTimer::stepEnd  ("BehaviorUpdate");


    if(d_schemeCorrection.getValue())
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



    if (!d_doCollisionsFirst.getValue())
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

    if(d_schemeCorrection.getValue())
        (*CP.getF())*=0.0;

    gaussSeidelConstraint(CP.getSize(), CP.getDfree()->ptr(), CP.getW()->lptr(), CP.getF()->ptr(), CP.getD()->ptr(), CP.getConstraintResolutions(), CP.getdF()->ptr());

    sofa::helper::AdvancedTimer::stepEnd  ("GaussSeidel");

    if (EMIT_EXTRA_DEBUG_MESSAGE)
        helper::afficheLCP(CP.getDfree()->ptr(), CP.getW()->lptr(), CP.getF()->ptr(),  CP.getSize());

    if ( d_displayTime.getValue() )
    {
        msg_info() << " Solve with GaussSeidel                " << ( (SReal) timer->getTime() - time)*timeScale<<" ms" ;
        time = (SReal) timer->getTime();
    }

    /// CORRECTIVE MOTION
    correctiveMotion(params, this->gnode);

    if ( d_displayTime.getValue() )
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

    if (!SOFA_NO_UPDATE_BBOX)
    {
        sofa::helper::AdvancedTimer::stepBegin("UpdateBBox");
        this->gnode->execute<UpdateBoundingBoxVisitor>(params);
        sofa::helper::AdvancedTimer::stepEnd("UpdateBBox");
    }
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Step");
#endif


}

void ConstraintAnimationLoop::computePredictiveForce(int dim, double* force, std::vector<core::behavior::ConstraintResolution*>& res)
{
    for(int i=0; i<dim; )
    {
        res[i]->initForce(i, force);
        i += res[i]->getNbLines();
    }
}

void ConstraintAnimationLoop::gaussSeidelConstraint(int dim, double* dfree, double** w, double* force,
        double* d, std::vector<ConstraintResolution*>& res, double* df=NULL)
{
    if(!dim)
        return;

    int iter, nb;
    double error=0.0;

    double tolerance = d_tol.getValue();
    int numItMax = d_maxIt.getValue();
    bool convergence = false;
    double sor = d_sor.getValue();
    bool allVerified = d_allVerified.getValue();
    sofa::helper::vector<double> tempForces;
    if(sor != 1.0) tempForces.resize(dim);

    if(d_scaleTolerance.getValue() && !allVerified)
        tolerance *= dim;

    for(int i=0; i<dim; )
    {
        res[i]->init(i, w, force);
        i += res[i]->getNbLines();
    }

    std::map < std::string, sofa::helper::vector<double> >* graphs = d_graphForces.beginEdit();
    graphs->clear();
    d_graphForces.endEdit();

    if(d_schemeCorrection.getValue())
    {
        std::cout<<"shemeCorrection => LCP before step 1"<<std::endl;
        helper::afficheLCP(dfree, w, force,  dim);
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

    sofa::helper::vector<double>& graph_residuals = (*d_graphErrors.beginEdit())["Error"];
    graph_residuals.clear();

    sofa::helper::vector<double> tabErrors;
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
                std::vector<double> errF(&force[j], &force[j+nb]);
                std::copy_n(&dfree[j], nb, &d[j]);

                //   (b) contribution of forces are added to d
                for(int k=0; k<dim; k++)
                    for(int l=0; l<nb; l++)
                        d[j+l] += w[j+l][k] * force[k];


                //3. the specific resolution of the constraint(s) is called
                res[j]->resolution(j, w, d, force, dfree);

                //4. the error is measured (displacement due to the new resolution (i.e. due to the new force))
                double contraintError = 0.0;
                if(nb > 1)
                {
                    for(int l=0; l<nb; l++)
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
        std::map < std::string, sofa::helper::vector<double> >* graphs = d_graphForces.beginEdit();
        for(int j=0; j<dim; j++)
        {
            std::ostringstream oss;
            oss << "f" << j;

            sofa::helper::vector<double>& graph_force = (*graphs)[oss.str()];
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

    sofa::helper::vector<double>& graph_constraints = (*d_graphConstraints.beginEdit())["Constraints"];
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

    double mu=0.8;
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


int ConstraintAnimationLoopClass = core::RegisterObject ( "Constraint animation loop manager" )
        .add< ConstraintAnimationLoop >()
        .addAlias("MasterConstraintSolver");

} // namespace animationloop

} // namespace component

} // namespace sofa
