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
#include <sofa/component/constraintset/LCPConstraintSolver.h>

#include <sofa/simulation/common/AnimateVisitor.h>
#include <sofa/simulation/common/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/SolveVisitor.h>
#include <sofa/simulation/common/BehaviorUpdatePositionVisitor.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/helper/LCPcalc.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/helper/system/thread/CTime.h>
#include <sofa/helper/AdvancedTimer.h>

#include <math.h>

#include <map>

namespace sofa
{

namespace component
{

namespace mastersolver
{

using namespace sofa::defaulttype;
using namespace helper::system::thread;
using namespace core::behavior;


ConstraintProblem::ConstraintProblem()
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
            _constraintsResolutions[j]->resolution(j, this->getW()->lptr(), this->getD()->ptr(), this->getF()->ptr());


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

    std::cout<<"------  No convergence in gaussSeidelConstraint Timed before time criterion !: error = " << error <<" ------" <<std::endl;
}


MasterConstraintSolver::MasterConstraintSolver()
    :displayTime(initData(&displayTime, false, "displayTime","Display time for each important step of MasterConstraintSolver.")),
     _tol( initData(&_tol, 0.00001, "tolerance", "Tolerance of the Gauss-Seidel")),
     _maxIt( initData(&_maxIt, 1000, "maxIterations", "Maximum number of iterations of the Gauss-Seidel")),
     doCollisionsFirst(initData(&doCollisionsFirst, false, "doCollisionsFirst","Compute the collisions first (to support penality-based contacts)")),
     doubleBuffer( initData(&doubleBuffer, false, "doubleBuffer","Buffer the constraint problem in a double buffer to be accessible with an other thread")),
     scaleTolerance( initData(&scaleTolerance, true, "scaleTolerance","Scale the error tolerance with the number of constraints")),
     _allVerified( initData(&_allVerified, false, "allVerified","All contraints must be verified (each constraint's error < tolerance)")),
     _sor( initData(&_sor, 1.0, "sor","Successive Over Relaxation parameter (0-2)")),
     schemeCorrection( initData(&schemeCorrection, false, "schemeCorrection","Apply new scheme where compliance is progressively corrected")),
     _graphErrors( initData(&_graphErrors,"graphErrors","Sum of the constraints' errors at each iteration")),
     _graphConstraints( initData(&_graphConstraints,"graphConstraints","Graph of each constraint's error at the end of the resolution"))
{
    bufCP1 = false;

    _graphErrors.setWidget("graph");
//	_graphErrors.setReadOnly(true);
    _graphErrors.setGroup("Graph");

    _graphConstraints.setWidget("graph");
//	_graphConstraints.setReadOnly(true);
    _graphConstraints.setGroup("Graph");

    CP1.clear(0,_tol.getValue());
    CP2.clear(0,_tol.getValue());

    timer = 0;
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
        serr<<"computeCollision is called"<<sendl;

    ////////////////// COLLISION DETECTION///////////////////////////////////////////////////////////////////////////////////////////
    sofa::helper::AdvancedTimer::stepBegin("Collision");
    computeCollision(params);
    sofa::helper::AdvancedTimer::stepEnd  ("Collision");
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if ( displayTime.getValue() )
    {
        sout<<" computeCollision " << ( (double) timer->getTime() - time)*timeScale <<" ms" <<sendl;
        time = (double) timer->getTime();
    }

}


void MasterConstraintSolver::freeMotion(simulation::Node *context, double &dt, const core::ExecParams* params)
{
    if (debug)
        serr<<"Free Motion is called"<<sendl;


    ///////////////////////////////////////////// FREE MOTION /////////////////////////////////////////////////////////////
    sofa::helper::AdvancedTimer::stepBegin("Free Motion");
    simulation::MechanicalBeginIntegrationVisitor(dt, params).execute(context);

    ////////////////// (optional) PREDICTIVE CONSTRAINT FORCES ///////////////////////////////////////////////////////////////////////////////////////////
    // When scheme Correction is used, the constraint forces computed at the previous time-step
    // are applied during the first motion, so which is no more a "free" motion but a "predictive" motion
    ///////////
    if(schemeCorrection.getValue())
    {
        for (unsigned int i=0; i<constraintCorrections.size(); i++ )
        {
            core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
            if (doubleBuffer.getValue() && bufCP1)
                cc->applyPredictiveConstraintForce(CP2.getF());
            else
                cc->applyPredictiveConstraintForce(CP1.getF());

        }
    }

    simulation::SolveVisitor(dt, true, params).execute(context);
    //simulation::MechanicalPropagateFreePositionVisitor().execute(context);
    {
        sofa::core::MechanicalParams mparams(*params);
        sofa::core::MultiVecCoordId xfree = sofa::core::VecCoordId::freePosition();
        mparams.x() = xfree;
        simulation::MechanicalPropagatePositionVisitor(0, xfree, true, &mparams ).execute(context);
    }
    sofa::helper::AdvancedTimer::stepEnd  ("Free Motion");

    //////// TODO : propagate velocity !!

    ////////propagate acceleration ? //////
    core::MultiVecDerivId dx_id = core::VecDerivId::dx();
    simulation::MechanicalVOpVisitor(dx_id, ConstVecId::null(), ConstVecId::null(), 1.0, params ).setMapped(true).execute(context);
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if ( displayTime.getValue() )
    {
        sout << ">>>>> Begin display MasterContactSolver time" << sendl;
        sout<<" Free Motion                           " << ( (double) timer->getTime() - time)*timeScale <<" ms" <<sendl;
        time = (double) timer->getTime();
    }
}

void MasterConstraintSolver::setConstraintEquations(simulation::Node *context, const core::ExecParams* params)
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
        writeAndAccumulateAndCountConstraintDirections(context, numConstraints, params);
    }

    core::MechanicalParams mparams = core::MechanicalParams(*params);
    simulation::MechanicalProjectJacobianMatrixVisitor(&mparams).execute(context);


    /// calling GetConstraintValueVisitor: each constraint provides its present violation
    /// for a given state (by default: free_position TODO: add VecId to make this method more generic)
    getIndividualConstraintViolations(context, params);

    if(!schemeCorrection.getValue())
    {
        /// calling getConstraintResolution: each constraint provides a method that is used to solve it during GS iterations
        getIndividualConstraintSolvingProcess(context, params);
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

void MasterConstraintSolver::writeAndAccumulateAndCountConstraintDirections(simulation::Node *context, unsigned int &numConstraints, const core::ExecParams* params)
{
    // calling resetConstraint on LMConstraints and MechanicalStates
    simulation::MechanicalResetConstraintVisitor(params).execute(context);

    core::ConstraintParams cparams = core::ConstraintParams(*params);
    cparams.setX(core::ConstVecCoordId::freePosition());
    cparams.setV(core::ConstVecDerivId::freeVelocity());

    // calling applyConstraint on each constraint
    MechanicalSetConstraint(core::MatrixDerivId::holonomicC(), numConstraints, &cparams).execute(context);
    sofa::helper::AdvancedTimer::valSet("numConstraints", numConstraints);

    // calling accumulateConstraint on the mappings
    MechanicalAccumulateConstraint2(core::MatrixDerivId::holonomicC(), &cparams).execute(context);

    if (debug)
        sout << "   1. resize constraints : numConstraints=" << numConstraints << sendl;

    if (doubleBuffer.getValue() && bufCP1)
        CP2.clear(numConstraints,this->_tol.getValue());
    else
        CP1.clear(numConstraints,this->_tol.getValue());
}

void MasterConstraintSolver::getIndividualConstraintViolations(simulation::Node *context, const core::ExecParams* params)
{
    if (debug)
        sout << "   2. compute violation" << sendl;

    core::ConstraintParams cparams = core::ConstraintParams(*params);
    cparams.setX(core::ConstVecCoordId::freePosition());
    cparams.setV(core::ConstVecDerivId::freeVelocity());

    if (doubleBuffer.getValue() && bufCP1)
    {
        constraintset::MechanicalGetConstraintValueVisitor(CP2.getDfree(), &cparams).execute(context);
    }
    else
    {
        constraintset::MechanicalGetConstraintValueVisitor(CP1.getDfree(), &cparams).execute(context);
    }
}

void MasterConstraintSolver::getIndividualConstraintSolvingProcess(simulation::Node *context, const core::ExecParams* params)
{
    /// calling getConstraintResolution: each constraint provides a method that is used to solve it during GS iterations
    if (debug)
        sout<<"   3. get resolution method for each constraint"<<sendl;

    if (doubleBuffer.getValue() && bufCP1)
        MechanicalGetConstraintResolutionVisitor(CP2.getConstraintResolutions(), 0, params).execute(context);
    else
        MechanicalGetConstraintResolutionVisitor(CP1.getConstraintResolutions(), 0, params).execute(context);
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
        if (doubleBuffer.getValue() && bufCP1)
            cc->getCompliance(CP2.getW());
        else
            cc->getCompliance(CP1.getW());
    }

    sofa::helper::AdvancedTimer::stepEnd  ("Get Compliance");

}

void MasterConstraintSolver::correctiveMotion(simulation::Node *context, const core::ExecParams* params)
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
            if (doubleBuffer.getValue() && bufCP1)
                cc->applyContactForce(CP2.getdF());
            else
                cc->applyContactForce(CP1.getdF());
        }
    }
    else
    {
        // ELSE => only correct the motion using F
        for (unsigned int i=0; i<constraintCorrections.size(); i++)
        {
            core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
            if (doubleBuffer.getValue() && bufCP1)
                cc->applyContactForce(CP2.getF());
            else
                cc->applyContactForce(CP1.getF());
        }
    }

    core::MechanicalParams mparams(*params);
    simulation::MechanicalPropagateAndAddDxVisitor(&mparams).execute(context);
    //simulation::MechanicalPropagatePositionAndVelocityVisitor().execute(context);


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

void MasterConstraintSolver::step ( double dt, const core::ExecParams* params )
{
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

    debug = this->f_printLog.getValue();

    if (debug)
        sout << "MasterConstraintSolver::step is called" << sendl;

    simulation::Node *context = dynamic_cast< simulation::Node * >(this->getContext());

    // This solver will work in freePosition and freeVelocity vectors.
    // We need to initialize them if it's not already done.
    simulation::MechanicalVInitVisitor<V_COORD>(VecCoordId::freePosition(), ConstVecCoordId::position()).execute(context);
    simulation::MechanicalVInitVisitor<V_DERIV>(VecDerivId::freeVelocity(), ConstVecDerivId::velocity()).execute(context);

    if (doCollisionsFirst.getValue())
    {
        /// COLLISION
        launchCollisionDetection(params);
    }

    // Update the BehaviorModels => to be removed ?
    // Required to allow the RayPickInteractor interaction
    sofa::helper::AdvancedTimer::stepBegin("BehaviorUpdate");
    simulation::BehaviorUpdatePositionVisitor(dt, params).execute(context);
    sofa::helper::AdvancedTimer::stepEnd  ("BehaviorUpdate");


    if(schemeCorrection.getValue())
    {
        // Compute the predictive force:
        numConstraints = 0;

        //1. Find the new constraint direction
        writeAndAccumulateAndCountConstraintDirections(context, numConstraints, params);

        //2. Get the constraint solving process:
        getIndividualConstraintSolvingProcess(context, params);

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
    freeMotion(context, dt, params);



    if (!doCollisionsFirst.getValue())
    {
        /// COLLISION
        launchCollisionDetection(params);
    }

    //////////////// BEFORE APPLYING CONSTRAINT  : propagate position through mapping
    core::MechanicalParams mparams(*params);
    simulation::MechanicalPropagatePositionVisitor(0, VecCoordId::position(), true, &mparams).execute(context);


    /// CONSTRAINT SPACE & COMPLIANCE COMPUTATION
    setConstraintEquations(context, params);

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
            sout << "Gauss-Seidel solver is called on problem of size" << CP2.getSize() << sendl;
        if(schemeCorrection.getValue())
            (*CP2.getF())*=0.0;

        gaussSeidelConstraint(CP2.getSize(), CP2.getDfree()->ptr(), CP2.getW()->lptr(), CP2.getF()->ptr(), CP2.getD()->ptr(), CP2.getConstraintResolutions(), CP2.getdF()->ptr());
    }
    else
    {
        if (debug)
            sout << "Gauss-Seidel solver is called on problem of size" << CP2.getSize() << sendl;
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
    correctiveMotion(context, params);

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

    simulation::MechanicalEndIntegrationVisitor endVisitor(dt, params);
    context->execute(&endVisitor);
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
            res[j]->resolution(j, w, d, force);

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

/////////////// debug //////////
//		if (i<10)
//		{
//			std::cerr<< std::setprecision(9)<<"FORCE and  DFREE at iteration "<<i<<": error"<<error<<std::endl;
//			for(k=0; k<dim; k++)
//				std::cerr<< std::setprecision(9) <<force[k]<<"     "<< dfree[k] <<std::endl;
//		}
////////////////////////////////

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

    if(!convergence)
        serr << "No convergence in gaussSeidelConstraint : error = " << error << sendl;
    else if ( displayTime.getValue() )
        sout<<" Convergence after " << i+1 << " iterations " << sendl;

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
