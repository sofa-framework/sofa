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
#include <sofa/component/mastersolver/MasterConstraintSolver.h>

#include <sofa/component/constraintset/ConstraintSolverImpl.h>

#include <sofa/simulation/common/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/SolveVisitor.h>
#include <sofa/simulation/common/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/common/MechanicalOperations.h>

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


namespace sofa
{

namespace component
{

namespace mastersolver
{

using namespace sofa::defaulttype;
using namespace helper::system::thread;
using namespace core::behavior;


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

    //	sout<<"------------------------------------ new iteration ---------------------------------"<<sendl;
    int i, j, k, l, nb;

    double errF[6];
    double error=0.0;


    bool convergence = false;

    double t0 = (double)_timer->getTime() ;
    double timeScale = 1.0 / (double)CTime::getTicksPerSec();

    /* // no init: the constraint problem has already been solved in the simulation...
    for(i=0; i<dim; )
    {
    	res[i]->init(i, w, force);
    	i += res[i]->nbLines;
    }
     */

    for(i=0; i<numItMax; i++)
    {
        error=0.0;
        for(j=0; j<_dim; ) // increment of j realized at the end of the loop
        {
            //std::cout<<" 1";
            //1. nbLines provide the dimension of the constraint  (max=6)
            //debug
            // int a=_constraintsResolutions.size();
            //std::cerr<<"&&"<<a<<"&&"<<std::endl;
            //end debug
            nb = _constraintsResolutions[j]->nbLines;


            //std::cout<<" 2.a ";
            //2. for each line we compute the actual value of d
            //   (a)d is set to dfree
            for(l=0; l<nb; l++)
            {
                errF[l] = _force[j+l];
                _d[j+l] = _dFree[j+l];
            }
            //std::cout<<" 2.b ";
            //   (b) contribution of forces are added to d
            for(k=0; k<_dim; k++)
                for(l=0; l<nb; l++)
                    _d[j+l] += _W[j+l][k] * _force[k];



            //3. the specific resolution of the constraint(s) is called
            //double** w = this->_W.ptr();
            //std::cout<<" 3 ";
            _constraintsResolutions[j]->resolution(j, this->getW()->lptr(), this->getD()->ptr(), this->getF()->ptr(), _dFree.ptr());


            //std::cout<<" 4 ";
            //4. the error is measured (displacement due to the new resolution (i.e. due to the new force))
            if(nb > 1)
            {
                double terr = 0.0, terr2;
                for(l=0; l<nb; l++)
                {
                    terr2=0;
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

        //std::cout<<" 5 ";

        /////////////////// GAUSS SEIDEL IS TIMED !!! /////////
        double t1 = (double)_timer->getTime();
        double dt = (t1 - t0)*timeScale;
        //std::cout<<"dt = "<<dt<<std::endl;
        if(dt > timeout)
        {
            return;
        }
        //std::cout<<" 6 ";
        ///////////////////////////////////////////////////////

        if(error < _tol*(_dim+1) && i>0) // do not stop at the first iteration (that is used for initial guess computation)
        {
            convergence = true;
            return;
        }
    }

    if (m_printLog)
    {
        std::cout << "------  No convergence in gaussSeidelConstraint Timed before time criterion !: error = "
                << error << " ------" << std::endl;
    }
}


MasterConstraintSolver::MasterConstraintSolver(simulation::Node* gnode)
    : Inherit(gnode)
    , displayTime(initData(&displayTime, false, "displayTime","Display time for each important step of MasterConstraintSolver."))
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

    std::cerr << "WARNING : MasterConstraintSolver is deprecated. Please use the combination of FreeMotionMasterSolver and GenericConstraintSolver." << std::endl;
}

MasterConstraintSolver::~MasterConstraintSolver()
{
    if (timer != 0)
    {
        delete timer;
        timer = 0;
    }
}

void MasterConstraintSolver::init()
{
    // Prevents ConstraintCorrection accumulation due to multiple MasterSolver initialization on dynamic components Add/Remove operations.
    if (!constraintCorrections.empty())
    {
        constraintCorrections.clear();
    }

    getContext()->get<core::behavior::BaseConstraintCorrection> ( &constraintCorrections, core::objectmodel::BaseContext::SearchDown );
}


void MasterConstraintSolver::launchCollisionDetection(const core::ExecParams* params)
{
    if (debug)
        sout<<"computeCollision is called"<<sendl;

    ////////////////// COLLISION DETECTION///////////////////////////////////////////////////////////////////////////////////////////
    sofa::helper::AdvancedTimer::stepBegin("Collision");
    computeCollision(params);
    sofa::helper::AdvancedTimer::stepEnd  ("Collision");
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if ( displayTime.getValue() )
    {
        sout<<" computeCollision                 " << ( (double) timer->getTime() - time)*timeScale <<" ms" <<sendl;
        time = (double) timer->getTime();
    }

}


void MasterConstraintSolver::freeMotion(const core::ExecParams* params /* PARAMS FIRST */, simulation::Node *context, double &dt)
{
    if (debug)
        sout<<"Free Motion is called"<<sendl;


    ///////////////////////////////////////////// FREE MOTION /////////////////////////////////////////////////////////////
    sofa::helper::AdvancedTimer::stepBegin("Free Motion");
    simulation::MechanicalBeginIntegrationVisitor(params /* PARAMS FIRST */, dt).execute(context);

    ////////////////// (optional) PREDICTIVE CONSTRAINT FORCES ///////////////////////////////////////////////////////////////////////////////////////////
    // When scheme Correction is used, the constraint forces computed at the previous time-step
    // are applied during the first motion, so which is no more a "free" motion but a "predictive" motion
    ///////////
    if(schemeCorrection.getValue())
    {
        sofa::core::ConstraintParams cparams(*params);
        sofa::core::MultiVecDerivId f =  core::VecDerivId::externalForce();

        for (unsigned int i=0; i<constraintCorrections.size(); i++ )
        {
            core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
            cc->applyPredictiveConstraintForce(&cparams, f, getCP()->getF());

            /*if (doubleBuffer.getValue() && bufCP1)
                cc->applyPredictiveConstraintForce(&cparams, f, CP2.getF());
            else
                cc->applyPredictiveConstraintForce(CP1.getF());*/

        }
    }

    simulation::SolveVisitor(params /* PARAMS FIRST */, dt, true).execute(context);

    {
        sofa::core::MechanicalParams mparams(*params);
        sofa::core::MultiVecCoordId xfree = sofa::core::VecCoordId::freePosition();
        mparams.x() = xfree;
        simulation::MechanicalPropagatePositionVisitor(&mparams /* PARAMS FIRST */, 0, xfree, true ).execute(context);
    }
    sofa::helper::AdvancedTimer::stepEnd  ("Free Motion");

    //////// TODO : propagate velocity !!

    ////////propagate acceleration ? //////

    //this is done to set dx to zero in subgraph
    core::MultiVecDerivId dx_id = core::VecDerivId::dx();
    simulation::MechanicalVOpVisitor(params /* PARAMS FIRST */, dx_id, core::ConstVecId::null(), core::ConstVecId::null(), 1.0 ).setMapped(true).execute(context);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if ( displayTime.getValue() )
    {
        sout << ">>>>> Begin display MasterContactSolver time" << sendl;
        sout<<" Free Motion                           " << ( (double) timer->getTime() - time)*timeScale <<" ms" <<sendl;
        time = (double) timer->getTime();
    }
}

void MasterConstraintSolver::setConstraintEquations(const core::ExecParams* params /* PARAMS FIRST */, simulation::Node *context)
{
    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetContactForce();
    }

    //////////////////////////////////////CONSTRAINTS RESOLUTION//////////////////////////////////////////////////////////////////////
    if (debug)
        sout<<"constraints Matrix construction is called"<<sendl;

    unsigned int numConstraints = 0;

    sofa::helper::AdvancedTimer::stepBegin("Constraints definition");


    if(!schemeCorrection.getValue())
    {
        /// calling resetConstraint & setConstraint & accumulateConstraint visitors
        /// and resize the constraint problem that will be solved
        writeAndAccumulateAndCountConstraintDirections(params /* PARAMS FIRST */, context, numConstraints);
    }


    core::MechanicalParams mparams = core::MechanicalParams(*params);
    simulation::MechanicalProjectJacobianMatrixVisitor(&mparams).execute(context);

    /// calling GetConstraintViolationVisitor: each constraint provides its present violation
    /// for a given state (by default: free_position TODO: add VecId to make this method more generic)
    getIndividualConstraintViolations(params /* PARAMS FIRST */, context);

    if(!schemeCorrection.getValue())
    {
        /// calling getConstraintResolution: each constraint provides a method that is used to solve it during GS iterations
        getIndividualConstraintSolvingProcess(params /* PARAMS FIRST */, context);
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

void MasterConstraintSolver::writeAndAccumulateAndCountConstraintDirections(const core::ExecParams* params /* PARAMS FIRST */, simulation::Node *context, unsigned int &numConstraints)
{
    // calling resetConstraint on LMConstraints and MechanicalStates
    simulation::MechanicalResetConstraintVisitor(params).execute(context);

    core::ConstraintParams cparams = core::ConstraintParams(*params);
    cparams.setX(core::ConstVecCoordId::freePosition());
    cparams.setV(core::ConstVecDerivId::freeVelocity());

    // calling applyConstraint on each constraint

    MechanicalSetConstraint(&cparams /* PARAMS FIRST */, core::MatrixDerivId::holonomicC(), numConstraints).execute(context);

    sofa::helper::AdvancedTimer::valSet("numConstraints", numConstraints);

    // calling accumulateConstraint on the mappings
    MechanicalAccumulateConstraint2(&cparams /* PARAMS FIRST */, core::MatrixDerivId::holonomicC()).execute(context);

    //if (debug)
    //    sout << "   1. resize constraints : numConstraints=" << numConstraints << sendl;

    getCP()->clear(numConstraints,this->_tol.getValue());

    /*if (doubleBuffer.getValue() && bufCP1)
        CP2.clear(numConstraints,this->_tol.getValue());
    else
        CP1.clear(numConstraints,this->_tol.getValue());*/
}

void MasterConstraintSolver::getIndividualConstraintViolations(const core::ExecParams* params /* PARAMS FIRST */, simulation::Node *context)
{
    //if (debug)
    //    sout << "   2. compute violation" << sendl;

    core::ConstraintParams cparams = core::ConstraintParams(*params);
    cparams.setX(core::ConstVecCoordId::freePosition());
    cparams.setV(core::ConstVecDerivId::freeVelocity());

    constraintset::MechanicalGetConstraintViolationVisitor(&cparams, getCP()->getDfree()).execute(context);
}

void MasterConstraintSolver::getIndividualConstraintSolvingProcess(const core::ExecParams* params /* PARAMS FIRST */, simulation::Node *context)
{
    /// calling getConstraintResolution: each constraint provides a method that is used to solve it during GS iterations
    //if (debug)
    //    sout<<"   3. get resolution method for each constraint"<<sendl;

    MechanicalGetConstraintResolutionVisitor(params, getCP()->getConstraintResolutions(), 0).execute(context);
}

void MasterConstraintSolver::computeComplianceInConstraintSpace()
{
    /// calling getCompliance => getDelassusOperator(_W) = H*C*Ht
    if (debug)
        sout<<"   4. get Compliance "<<sendl;

    sofa::helper::AdvancedTimer::stepBegin("Get Compliance");
    for (unsigned int i=0; i<constraintCorrections.size(); i++ )
    {
        core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->addComplianceInConstraintSpace(core::ConstraintParams::defaultInstance(), getCP()->getW());

        /*if (doubleBuffer.getValue() && bufCP1)
            cc->getCompliance(CP2.getW());
        else
            cc->getCompliance(CP1.getW());*/
    }

    sofa::helper::AdvancedTimer::stepEnd  ("Get Compliance");

}

void MasterConstraintSolver::correctiveMotion(const core::ExecParams* params /* PARAMS FIRST */, simulation::Node *context)
{

    if (debug)
        sout<<"constraintCorrections motion is called"<<sendl;

    sofa::helper::AdvancedTimer::stepBegin("Corrective Motion");

    if(schemeCorrection.getValue())
    {
        // IF SCHEME CORRECTIVE=> correct the motion using dF
        for (unsigned int i=0; i<constraintCorrections.size(); i++)
        {
            core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
            cc->applyContactForce(getCP()->getdF());

            /*if (doubleBuffer.getValue() && bufCP1)
                cc->applyContactForce(CP2.getdF());
            else
                cc->applyContactForce(CP1.getdF());*/
        }
    }
    else
    {
        // ELSE => only correct the motion using F
        for (unsigned int i=0; i<constraintCorrections.size(); i++)
        {
            core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
            cc->applyContactForce(getCP()->getF());

            /*if (doubleBuffer.getValue() && bufCP1)
                cc->applyContactForce(CP2.getF());
            else
                cc->applyContactForce(CP1.getF());*/
        }
    }

    simulation::common::MechanicalOperations mop(params, this->getContext());

    mop.propagateV(core::VecDerivId::velocity());

    mop.propagateDx(core::VecDerivId::dx());

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

void MasterConstraintSolver::step( const core::ExecParams* params /* PARAMS FIRST */, double dt )
{
    sofa::helper::AdvancedTimer::stepBegin("MasterSolverStep");

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


        //////////////////////////////////////////////////////////////////
        time = 0.0;
        double totaltime = 0.0;
        timeScale = 1.0 / (double)CTime::getTicksPerSec() * 1000;
        if ( displayTime.getValue() )
        {
            if (timer == 0)
                timer = new CTime();

            time = (double) timer->getTime();
            totaltime = time;
            sout<<sendl;
        }
        if (doubleBuffer.getValue())
        {
            // SWAP BUFFER:
            bufCP1 = !bufCP1;
        }

#ifndef WIN32
        if (_realTimeCompensation.getValue())
        {
            if (timer == 0)
            {
                timer = new CTime();
                compTime = iterationTime = (double)timer->getTime();
            }
            else
            {
                double actTime = double(timer->getTime());
                double compTimeDiff = actTime - compTime;
                double iterationTimeDiff = actTime - iterationTime;
                iterationTime = actTime;
                std::cout << "Total time = " << iterationTimeDiff << std::endl;
                int toSleep = (int)floor(dt*1000000-compTimeDiff);
                //std::cout << "To sleep: " << toSleep << std::endl;
                if (toSleep > 0)
                    usleep(toSleep);
                else
                    serr << "Cannot achieve frequency for dt = " << dt << sendl;
                compTime = (double)timer->getTime();
            }
        }
#endif

        debug = this->f_printLog.getValue();

        if (debug)
            sout << "MasterConstraintSolver::step is called" << sendl;


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
        simulation::BehaviorUpdatePositionVisitor(params /* PARAMS FIRST */, dt).execute(this->gnode);
        sofa::helper::AdvancedTimer::stepEnd  ("BehaviorUpdate");


        if(schemeCorrection.getValue())
        {
            // Compute the predictive force:
            numConstraints = 0;

            //1. Find the new constraint direction
            writeAndAccumulateAndCountConstraintDirections(params /* PARAMS FIRST */, this->gnode, numConstraints);

            //2. Get the constraint solving process:
            getIndividualConstraintSolvingProcess(params /* PARAMS FIRST */, this->gnode);

            //3. Use the stored forces to compute
            if (debug)
            {
                if (doubleBuffer.getValue() && bufCP1)
                {
                    computePredictiveForce(CP2.getSize(), CP2.getF()->ptr(), CP2.getConstraintResolutions());
                    std::cout << "getF() after computePredictiveForce:" << std::endl;
                    helper::afficheResult(CP2.getF()->ptr(),CP2.getSize());
                }
                else
                {
                    computePredictiveForce(CP1.getSize(), CP1.getF()->ptr(), CP1.getConstraintResolutions());
                    std::cout << "getF() after computePredictiveForce:" << std::endl;
                    helper::afficheResult(CP1.getF()->ptr(),CP1.getSize());
                }
            }
        }

        if (debug)
        {
            if (doubleBuffer.getValue() && bufCP1)
            {
                (*CP2.getF())*=0.0;
                computePredictiveForce(CP2.getSize(), CP2.getF()->ptr(), CP2.getConstraintResolutions());
                std::cout << "getF() after re-computePredictiveForce:" << std::endl;
                helper::afficheResult(CP2.getF()->ptr(),CP2.getSize());
            }
            else
            {
                (*CP1.getF())*=0.0;
                computePredictiveForce(CP1.getSize(), CP1.getF()->ptr(), CP1.getConstraintResolutions());
                std::cout << "getF() after re-computePredictiveForce:" << std::endl;
                helper::afficheResult(CP1.getF()->ptr(),CP1.getSize());
            }
        }




        /// FREE MOTION
        freeMotion(params /* PARAMS FIRST */, this->gnode, dt);



        if (!doCollisionsFirst.getValue())
        {
            /// COLLISION
            launchCollisionDetection(params);
        }

        //////////////// BEFORE APPLYING CONSTRAINT  : propagate position through mapping
        core::MechanicalParams mparams(*params);
        simulation::MechanicalPropagatePositionVisitor(&mparams /* PARAMS FIRST */, 0, core::VecCoordId::position(), true).execute(this->gnode);


        /// CONSTRAINT SPACE & COMPLIANCE COMPUTATION
        setConstraintEquations(params /* PARAMS FIRST */, this->gnode);

        if (debug)
        {
            if (doubleBuffer.getValue() && bufCP1)
            {
                std::cout << "getF() after setConstraintEquations:" << std::endl;
                helper::afficheResult(CP2.getF()->ptr(),CP2.getSize());
            }
            else
            {
                std::cout << "getF() after setConstraintEquations:" << std::endl;
                helper::afficheResult(CP1.getF()->ptr(),CP1.getSize());
            }
        }

        sofa::helper::AdvancedTimer::stepBegin("GaussSeidel");

        if (doubleBuffer.getValue() && bufCP1)
        {
            if (debug)
                sout << "Gauss-Seidel solver is called on problem of size " << CP2.getSize() << sendl;
            if(schemeCorrection.getValue())
                (*CP2.getF())*=0.0;

            gaussSeidelConstraint(CP2.getSize(), CP2.getDfree()->ptr(), CP2.getW()->lptr(), CP2.getF()->ptr(), CP2.getD()->ptr(), CP2.getConstraintResolutions(), CP2.getdF()->ptr());
        }
        else
        {
            if (debug)
                sout << "Gauss-Seidel solver is called on problem of size " << CP1.getSize() << sendl;
            if(schemeCorrection.getValue())
                (*CP1.getF())*=0.0;

            gaussSeidelConstraint(CP1.getSize(), CP1.getDfree()->ptr(), CP1.getW()->lptr(), CP1.getF()->ptr(), CP1.getD()->ptr(), CP1.getConstraintResolutions(), CP1.getdF()->ptr());
        }

        sofa::helper::AdvancedTimer::stepEnd  ("GaussSeidel");

        if (debug)
        {
            if (doubleBuffer.getValue() && bufCP1)
                helper::afficheLCP(CP2.getDfree()->ptr(), CP2.getW()->lptr(), CP2.getF()->ptr(),  CP2.getSize());
            else
                helper::afficheLCP(CP1.getDfree()->ptr(), CP1.getW()->lptr(), CP1.getF()->ptr(),  CP1.getSize());
        }

        if ( displayTime.getValue() )
        {
            sout << " Solve with GaussSeidel                " << ( (double) timer->getTime() - time)*timeScale<<" ms" <<sendl;
            time = (double) timer->getTime();
        }

        /// CORRECTIVE MOTION
        correctiveMotion(params /* PARAMS FIRST */, this->gnode);
        //       if (doubleBuffer.getValue() && bufCP1)
        //           std::cout << " #C: " << CP2.getSize() << " constraints" << std::endl;
        //       else
        //           std::cout << " #C: " << CP1.getSize() << " constraints" << std::endl;


        if ( displayTime.getValue() )
        {
            sout << " ContactCorrections                    " << ( (double) timer->getTime() - time)*timeScale <<" ms" <<sendl;
            sout << "  = Total                              " << ( (double) timer->getTime() - totaltime)*timeScale <<" ms" <<sendl;
            if (doubleBuffer.getValue() && bufCP1)
                sout << " With : " << CP2.getSize() << " constraints" << sendl;
            else
                sout << " With : " << CP1.getSize() << " constraints" << sendl;

            sout << "<<<<< End display MasterContactSolver time." << sendl;
        }

        simulation::MechanicalEndIntegrationVisitor endVisitor(params /* PARAMS FIRST */, dt);
        this->gnode->execute(&endVisitor);

        ////////////////////////////////////////////////////////////////////////////////////////////////

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

    //////////////////////////////////////////////////////////////////////
#ifndef  DEPRECATED_MASTERSOLVER
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
    simulation::Visitor::printCloseNode(std::string("Step"));
#endif
    nbSteps.setValue(nbSteps.getValue() + 1);
#endif//  DEPRECATED_MASTERSOLVER
    /////////////////////////////////////////////////////////////////////

    sofa::helper::AdvancedTimer::stepEnd("MasterSolverStep");
}

void MasterConstraintSolver::computePredictiveForce(int dim, double* force, std::vector<core::behavior::ConstraintResolution*>& res)
{
    for(int i=0; i<dim; )
    {
        res[i]->initForce(i, force);
        i += res[i]->nbLines;
    }
}

void MasterConstraintSolver::gaussSeidelConstraint(int dim, double* dfree, double** w, double* force,
        double* d, std::vector<ConstraintResolution*>& res, double* df=NULL)
{
    if(!dim)
        return;

    //std::cout << "Dim = " << dim << std::endl;

    int i, j, k, l, nb;

    double errF[6];
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
    /*	for(j=0; j<dim; j++)
    {
    	std::ostringstream oss;
    	oss << "f" << j;

    	sofa::helper::vector<double>& graph_force = (*graphs)[oss.str()];
    	graph_force.clear();
    }	*/
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
            //std::cout << "dim = " << nb << std::endl;

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

                ///////////// debug //////////
                /*		if (i<3 && j<3)
                {
                	std::cerr<<".............. iteration "<<i<< std::endl;
                	std::cerr<<"d ["<<j<<"]="<<d[j]<<"  - d ["<<j+1<<"]="<<d[j+1]<<"  - d ["<<j+2<<"]="<<d[j+2]<<std::endl;
                }*/
                //////////////////////////////

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
                if (i==0) printf("ERROR : constraint %d has a compliance equal to zero on the diagonal\n",j);
                j += nb;
            }
        }


        //for(k=0; k<dim; k++)
        //     std::cout << "F1[" << k << "] = " <<  force[k]<<"     "<< dfree[k] <<std::endl;
        //std::cout << "Iter " << i << ", err = " << error << std::endl;


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

    if (debug)
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
        //delete res[i];  // do it in the "clear function" of the constraint problem: the constraint problem can be put in a buffer
        //res[i] = NULL;
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




void MasterConstraintSolver::debugWithContact(int numConstraints)
{

    double mu=0.8;
    if (doubleBuffer.getValue() && bufCP1)
    {
        helper::nlcp_gaussseidel(numConstraints, CP2.getDfree()->ptr(), CP2.getW()->lptr(), CP2.getF()->ptr(), mu, _tol.getValue(), _maxIt.getValue(), false, debug);
        CP2.getF()->clear();
        CP2.getF()->resize(numConstraints);
    }
    else
    {
        helper::nlcp_gaussseidel(numConstraints, CP1.getDfree()->ptr(), CP1.getW()->lptr(), CP1.getF()->ptr(), mu, _tol.getValue(), _maxIt.getValue(), false, debug);
        CP1.getF()->clear();
        CP1.getF()->resize(numConstraints);
    }

}


SOFA_DECL_CLASS ( MasterConstraintSolver )

int MasterConstraintSolverClass = core::RegisterObject ( "Constraint solver" )
        .add< MasterConstraintSolver >()
        ;

} // namespace odesolver

} // namespace component

} // namespace sofa
