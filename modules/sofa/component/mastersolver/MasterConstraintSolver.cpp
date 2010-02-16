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
#include <sofa/component/constraint/LCPConstraintSolver.h>

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

using namespace sofa::component::odesolver;
using namespace sofa::defaulttype;
using namespace helper::system::thread;
using namespace core::componentmodel::behavior;


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
     _graphErrors( initData(&_graphErrors,"graphErrors","Sum of the constraints' errors at each iteration")),
     _graphConstraints( initData(&_graphConstraints,"graphConstraints","Graph of each constraint's error at the end of the resolution"))
{
    bufCP1 = true;

    _graphErrors.setWidget("graph");
    _graphErrors.setReadOnly(true);
    _graphErrors.setGroup("Graph");

    _graphConstraints.setWidget("graph");
    _graphConstraints.setReadOnly(true);
    _graphConstraints.setGroup("Graph");
}

MasterConstraintSolver::~MasterConstraintSolver()
{
}

void MasterConstraintSolver::init()
{
    // Prevents ConstraintCorrection accumulation due to multiple MasterSolver initialization on dynamic components Add/Remove operations.
    if (!constraintCorrections.empty())
    {
        constraintCorrections.clear();
    }

    getContext()->get<core::componentmodel::behavior::BaseConstraintCorrection> ( &constraintCorrections, core::objectmodel::BaseContext::SearchDown );
}

void MasterConstraintSolver::step ( double dt )
{
    CTime *timer;
    double time = 0.0, totaltime = 0.0;
    double timeScale = 1.0 / (double)CTime::getTicksPerSec() * 1000;
    if ( displayTime.getValue() )
    {
        timer = new CTime();
        time = (double) timer->getTime();
        totaltime = time;
        sout<<sendl;
    }

    bool debug =this->f_printLog.getValue();
    if (debug)
        serr<<"MasterConstraintSolver::step is called"<<sendl;
    simulation::Node *context = dynamic_cast<simulation::Node *>(this->getContext()); // access to current node


    if (doCollisionsFirst.getValue())
    {
        if (debug)
            serr<<"computeCollision is called"<<sendl;

        ////////////////// COLLISION DETECTION///////////////////////////////////////////////////////////////////////////////////////////
        sofa::helper::AdvancedTimer::stepBegin("Collision");
        computeCollision();
        sofa::helper::AdvancedTimer::stepEnd  ("Collision");
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if ( displayTime.getValue() )
        {
            sout<<" computeCollision " << ( (double) timer->getTime() - time)*timeScale <<" ms" <<sendl;
            time = (double) timer->getTime();
        }
    }

    // Update the BehaviorModels
    // Required to allow the RayPickInteractor interaction
    sofa::helper::AdvancedTimer::stepBegin("BehaviorUpdate");
    simulation::BehaviorUpdatePositionVisitor(dt).execute(context);
    sofa::helper::AdvancedTimer::stepEnd  ("BehaviorUpdate");
    if (debug)
        serr<<"Free Motion is called"<<sendl;

    ///////////////////////////////////////////// FREE MOTION /////////////////////////////////////////////////////////////
    sofa::helper::AdvancedTimer::stepBegin("Free Motion");
    simulation::MechanicalBeginIntegrationVisitor(dt).execute(context);
    simulation::SolveVisitor(dt, true).execute(context);
    simulation::MechanicalPropagateFreePositionVisitor().execute(context);
    sofa::helper::AdvancedTimer::stepEnd  ("Free Motion");

    //////// TODO : propagate velocity !!

    ////////propagate acceleration ? //////
    core::componentmodel::behavior::BaseMechanicalState::VecId dx_id = core::componentmodel::behavior::BaseMechanicalState::VecId::dx();
    simulation::MechanicalVOpVisitor(dx_id).execute(context);
    simulation::MechanicalPropagateDxVisitor(dx_id,true).execute(context); //ignore mask here (is it necessary?)
    simulation::MechanicalVOpVisitor(dx_id).execute(context);
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if ( displayTime.getValue() )
    {
        sout << ">>>>> Begin display MasterContactSolver time" << sendl;
        sout<<" Free Motion                           " << ( (double) timer->getTime() - time)*timeScale <<" ms" <<sendl;
        time = (double) timer->getTime();
    }

    if (!doCollisionsFirst.getValue())
    {
        if (debug)
            serr<<"computeCollision is called"<<sendl;

        ////////////////// COLLISION DETECTION///////////////////////////////////////////////////////////////////////////////////////////
        sofa::helper::AdvancedTimer::stepBegin("Collision");
        computeCollision();
        sofa::helper::AdvancedTimer::stepEnd  ("Collision");
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if ( displayTime.getValue() )
        {
            sout<<" ComputeCollision                      " << ( (double) timer->getTime() - time)*timeScale <<" ms" <<sendl;
            time = (double) timer->getTime();
        }
    }

    //////////////// BEFORE APPLYING CONSTRAINT  : propagate position through mapping
    simulation::MechanicalPropagatePositionVisitor().execute(context);

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetContactForce();
    }

    //////////////////////////////////////CONSTRAINTS RESOLUTION//////////////////////////////////////////////////////////////////////
    if (debug)
        serr<<"constraints Matrix construction is called"<<sendl;

    unsigned int numConstraints = 0;


    sofa::helper::AdvancedTimer::stepBegin("Constraints definition");
    // calling resetConstraint on LMConstraints and MechanicalStates
    simulation::MechanicalResetConstraintVisitor().execute(context);

    // calling applyConstraint on each constraint
    MechanicalSetConstraint(numConstraints).execute(context);
    sofa::helper::AdvancedTimer::valSet("numConstraints", numConstraints);

    // calling accumulateConstraint on the mappings
    MechanicalAccumulateConstraint2().execute(context);

    if (debug)
        serr<<"   1. resize constraints : numConstraints="<< numConstraints<<sendl;

    if (doubleBuffer.getValue() && bufCP1)
        CP2.clear(numConstraints,this->_tol.getValue());
    else
        CP1.clear(numConstraints,this->_tol.getValue());

    if (debug)
        serr<<"   2. compute violation"<<sendl;
    // calling getConstraintValue
    if (doubleBuffer.getValue() && bufCP1)
        constraint::MechanicalGetConstraintValueVisitor(CP2.getDfree()).execute(context);
    else
        constraint::MechanicalGetConstraintValueVisitor(CP1.getDfree()).execute(context);

    /// calling getConstraintResolution: each constraint provides a method that is used to solve it during GS iterations
    if (debug)
        serr<<"   3. get resolution method for each constraint"<<sendl;

    if (doubleBuffer.getValue() && bufCP1)
        MechanicalGetConstraintResolutionVisitor(CP2.getConstraintResolutions()).execute(context);
    else
        MechanicalGetConstraintResolutionVisitor(CP1.getConstraintResolutions()).execute(context);
    sofa::helper::AdvancedTimer::stepEnd  ("Constraints definition");

    /// calling getCompliance => getDelassusOperator(_W) = H*C*Ht
    if (debug)
        serr<<"   4. get Compliance "<<sendl;

    sofa::helper::AdvancedTimer::stepBegin("Get Compliance");
    for (unsigned int i=0; i<constraintCorrections.size(); i++ )
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        if (doubleBuffer.getValue() && bufCP1)
            cc->getCompliance(CP2.getW());
        else
            cc->getCompliance(CP1.getW());
    }

    sofa::helper::AdvancedTimer::stepEnd  ("Get Compliance");

    if ( displayTime.getValue() )
    {
        sout<<" Build problem in the constraint space " << ( (double) timer->getTime() - time)*timeScale<<" ms" <<sendl;
        time = (double) timer->getTime();
    }

/////////////////////////////////// debug with contact: compare results when only contacts are involved in the scene ////////////////
    ///   TO BE REMOVED  ///
    sofa::helper::AdvancedTimer::stepBegin("GaussSeidel");
    bool debugWithContact = false;
    if (debugWithContact)
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
    ///  END TO BE REMOVED  ///
///////////////////////////////////////////////////

    if (debug)
        serr<<"Gauss-Seidel solver is called on problem of size"<<numConstraints<<sendl;
    if (doubleBuffer.getValue() && bufCP1)
        gaussSeidelConstraint(numConstraints, CP2.getDfree()->ptr(), CP2.getW()->lptr(), CP2.getF()->ptr(), CP2.getD()->ptr(), CP2.getConstraintResolutions());
    else
        gaussSeidelConstraint(numConstraints, CP1.getDfree()->ptr(), CP1.getW()->lptr(), CP1.getF()->ptr(), CP1.getD()->ptr(), CP1.getConstraintResolutions());

    sofa::helper::AdvancedTimer::stepEnd  ("GaussSeidel");

    if (debug)
    {
        if (doubleBuffer.getValue() && bufCP1)
            helper::afficheLCP(CP2.getDfree()->ptr(), CP2.getW()->lptr(), CP2.getF()->ptr(),  numConstraints);
        else
            helper::afficheLCP(CP1.getDfree()->ptr(), CP1.getW()->lptr(), CP1.getF()->ptr(),  numConstraints);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if ( displayTime.getValue() )
    {
        sout<<" Solve with GaussSeidel                " <<( (double) timer->getTime() - time)*timeScale<<" ms" <<sendl;
        time = (double) timer->getTime();
    }

//

    if (debug)
        sout<<"constraintCorrections motion is called"<<sendl;

    ///////////////////////////////////////CORRECTIVE MOTION //////////////////////////////////////////////////////////////////////////
    sofa::helper::AdvancedTimer::stepBegin("Corrective Motion");
    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        if (doubleBuffer.getValue() && bufCP1)
            cc->applyContactForce(CP2.getF());
        else
            cc->applyContactForce(CP1.getF());
    }
    simulation::MechanicalPropagateAndAddDxVisitor().execute(context);
    //simulation::MechanicalPropagatePositionAndVelocityVisitor().execute(context);

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::componentmodel::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetContactForce();
    }
    sofa::helper::AdvancedTimer::stepEnd ("Corrective Motion");

    if ( displayTime.getValue() )
    {
        sout<<" ContactCorrections                    " <<( (double) timer->getTime() - time)*timeScale <<" ms" <<sendl;
        sout<<"  = Total                              " <<( (double) timer->getTime() - totaltime)*timeScale <<" ms" <<sendl;
        sout<<" With : " << numConstraints << " constraints" << sendl;
        sout << "<<<<< End display MasterContactSolver time." << sendl;
    }

    simulation::MechanicalEndIntegrationVisitor endVisitor(dt);
    context->execute(&endVisitor);
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    ////// in case of a double buffer we exchange buffer 1 and buffer 2: ////

    //////// debug double buffer
    //bool debugWithDoubleBuffer = true;

    //if(debugWithDoubleBuffer)
    //{
    //	double timeOut = 0.001;  //1ms

    //	if (this->getConstraintProblem() != NULL)
    //	{
    //		//std::cout<<"new gauss Seidel Constraint timed called on the buf problem of size"<<this->getConstraintProblem()->getSize()<<std::endl;
    //		//double before = (double) timer->getTime() ;
    //		this->getConstraintProblem()->gaussSeidelConstraintTimed(timeOut, 1000);
    //		//double after = (double) timer->getTime() ;
    //		//std::cout<<"gauss Seidel Constraint answers in  "<<(after-before)*timeScale<<"  Msec"<<std::endl;
    //	}
    //	else
    //		std::cout<<"this->getConstraintProblem() is null"<<std::endl;

    //}
    //////////////////////////////


    if (doubleBuffer.getValue())
    {
        /// test:

        //std::cout<<"swap Buffer: size new ConstraintProblem = "<<numConstraints<< " size old buf Problem "<<getConstraintProblem()->getSize()<<std::endl;
        bufCP1 = !bufCP1;
// 		int a=getConstraintProblem()->getConstraintResolutions().size();
        //std::cerr<<"##"<<a<<"##"<<std::endl;
    }
}

void MasterConstraintSolver::gaussSeidelConstraint(int dim, double* dfree, double** w, double* force,
        double* d, std::vector<ConstraintResolution*>& res)
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

    _graphErrors.endEdit();

    sofa::helper::vector<double>& graph_constraints = (*_graphConstraints.beginEdit())["Constraints"];
    graph_constraints.clear();

    for(j=0; j<dim; )
    {
        nb = res[j]->nbLines;

        if(tabErrors[j])
            graph_constraints.push_back(tabErrors[j]);

        j += nb;
    }

    _graphConstraints.endEdit();
}


SOFA_DECL_CLASS ( MasterConstraintSolver )

int MasterConstraintSolverClass = core::RegisterObject ( "Constraint solver" )
        .add< MasterConstraintSolver >()
        ;

} // namespace odesolver

} // namespace component

} // namespace sofa
