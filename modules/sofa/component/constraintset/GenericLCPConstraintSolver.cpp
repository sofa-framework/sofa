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

#include <sofa/component/constraintset/GenericLCPConstraintSolver.h>

#include <sofa/simulation/common/AnimateVisitor.h>
#include <sofa/simulation/common/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/SolveVisitor.h>

#include <sofa/simulation/common/Simulation.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/gl/Cylinder.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/system/thread/CTime.h>
#include <math.h>
#include <iostream>

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace constraintset
{


#define MAX_NUM_CONSTRAINTS 3000
//#define DISPLAY_TIME

GenericLCPConstraintSolver::GenericLCPConstraintSolver()
    : displayTime(initData(&displayTime, false, "displayTime","Display time for each important step of GenericLCPConstraintSolver."))
    , maxIt( initData(&maxIt, 1000, "maxIterations", "maximal number of iterations of the Gauss-Seidel algorithm"))
    , tolerance( initData(&tolerance, 0.001, "tolerance", "residual error threshold for termination of the Gauss-Seidel algorithm"))
    , sor( initData(&sor, 1.0, "sor", "Successive Over Relaxation parameter (0-2)"))
    , scaleTolerance( initData(&scaleTolerance, true, "scaleTolerance", "Scale the error tolerance with the number of constraints"))
    , allVerified( initData(&allVerified, false, "allVerified", "All contraints must be verified (each constraint's error < tolerance)"))
    , schemeCorrection( initData(&schemeCorrection, false, "schemeCorrection", "Apply new scheme where compliance is progressively corrected"))
    , graphErrors( initData(&graphErrors,"graphErrors","Sum of the constraints' errors at each iteration"))
    , graphConstraints( initData(&graphConstraints,"graphConstraints","Graph of each constraint's error at the end of the resolution"))
//, graphForces( initData(&graphForces,"graphForces","Graph of each constraint's force at each step of the resolution"))
    , lcp(&lcp1)
    , last_lcp(NULL)
{
    addAlias(&maxIt, "maxIt");

    graphErrors.setWidget("graph");
    graphErrors.setGroup("Graph");

    graphConstraints.setWidget("graph");
    graphConstraints.setGroup("Graph");

//	graphForces.setWidget("graph");
//	graphForces.setGroup("Graph2");
}

GenericLCPConstraintSolver::~GenericLCPConstraintSolver()
{
    lcp1.freeConstraintResolutions();
    lcp2.freeConstraintResolutions();
    lcp3.freeConstraintResolutions();
}

void GenericLCPConstraintSolver::init()
{
    core::behavior::ConstraintSolver::init();

    // Prevents ConstraintCorrection accumulation due to multiple MasterSolver initialization on dynamic components Add/Remove operations.
    if (!constraintCorrections.empty())
    {
        constraintCorrections.clear();
    }

    getContext()->get<core::behavior::BaseConstraintCorrection>(&constraintCorrections, core::objectmodel::BaseContext::SearchDown);

    context = (simulation::Node*) getContext();
}

bool GenericLCPConstraintSolver::prepareStates(double /*dt*/, MultiVecId id, core::ConstraintParams::ConstOrder)
{
    if (id.getDefaultId() != VecId::freePosition()) return false;
    sofa::helper::AdvancedTimer::StepVar vtimer("PrepareStates");

    last_lcp = lcp;
    simulation::MechanicalVOpVisitor((VecId)core::VecDerivId::dx()).setMapped(true).execute( context); //dX=0
    //simulation::MechanicalPropagateDxVisitor(dx_id,true,true).execute( context); //Propagate dX //ignore the mask here

    if( f_printLog.getValue())
        serr<<" propagate DXn performed - collision called"<<sendl;

    time = 0.0;
    timeTotal=0.0;
    timeScale = 1000.0 / (double)CTime::getTicksPerSec();

    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetContactForce();
    }

    if ( displayTime.getValue() )
    {
        time = (double) timer.getTime();
        timeTotal = (double) timerTotal.getTime();
    }
    return true;
}

bool GenericLCPConstraintSolver::buildSystem(double /*dt*/, MultiVecId, core::ConstraintParams::ConstOrder)
{
    unsigned int numConstraints = 0;
    core::ConstraintParams cparams;

    cparams.setX(core::ConstVecCoordId::freePosition());
    cparams.setV(core::ConstVecDerivId::freeVelocity());

    sofa::helper::AdvancedTimer::stepBegin("Accumulate Constraint");
    // mechanical action executed from root node to propagate the constraints
    simulation::MechanicalResetConstraintVisitor(&cparams).execute(context);
    // calling buildConstraintMatrix
    simulation::MechanicalAccumulateConstraint(core::MatrixDerivId::holonomicC(), numConstraints, &cparams).execute(context);
    sofa::helper::AdvancedTimer::stepEnd  ("Accumulate Constraint");
    sofa::helper::AdvancedTimer::valSet("numConstraints", numConstraints);

    lcp->setNbConst(numConstraints);

    sofa::helper::AdvancedTimer::stepBegin("Get Constraint Value");
    MechanicalGetConstraintValueVisitor(&lcp->dFree, &cparams).execute(context);
    sofa::helper::AdvancedTimer::stepEnd ("Get Constraint Value");

    sofa::helper::AdvancedTimer::stepBegin("Get Constraint Resolutions");
    MechanicalGetConstraintResolutionVisitor(lcp->constraintsResolutions, &cparams).execute(context);
    sofa::helper::AdvancedTimer::stepEnd("Get Constraint Resolutions");

    if (this->f_printLog.getValue()) sout<<"GenericLCPConstraintSolver: "<<numConstraints<<" constraints"<<sendl;

    sofa::helper::AdvancedTimer::stepBegin("Get Compliance");
    if (this->f_printLog.getValue()) sout<<" computeCompliance in "  << constraintCorrections.size()<< " constraintCorrections" <<sendl;
    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->getCompliance(&lcp->W);
    }
    sofa::helper::AdvancedTimer::stepEnd  ("Get Compliance");
    if (this->f_printLog.getValue()) sout<<" computeCompliance_done "  <<sendl;

    if ( displayTime.getValue() )
    {
        sout<<" build_LCP " << ( (double) timer.getTime() - time)*timeScale<<" ms" <<sendl;
        time = (double) timer.getTime();
    }
    return true;
}

bool GenericLCPConstraintSolver::solveSystem(double /*dt*/, MultiVecId, core::ConstraintParams::ConstOrder)
{
    lcp->setTolerance(tolerance.getValue());
    lcp->setMaxIter(maxIt.getValue());
    lcp->scaleTolerance = scaleTolerance.getValue();
    lcp->allVerified = allVerified.getValue();
    lcp->sor = sor.getValue();

    sofa::helper::AdvancedTimer::stepBegin("ConstraintsGaussSeidel");
    lcp->gaussSeidel(0, this);
    sofa::helper::AdvancedTimer::stepEnd("ConstraintsGaussSeidel");

    if ( displayTime.getValue() )
    {
        sout<<" TOTAL solve_LCP " <<( (double) timer.getTime() - time)*timeScale<<" ms" <<sendl;
        time = (double) timer.getTime();
    }

    return true;
}

bool GenericLCPConstraintSolver::applyCorrection(double /*dt*/, MultiVecId , core::ConstraintParams::ConstOrder)
{
    if(this->f_printLog.getValue())
        serr<<"keepContactForces done"<<sendl;

    sofa::helper::AdvancedTimer::stepBegin("Apply Contact Force");
    //	MechanicalApplyContactForceVisitor(_result).execute(context);
    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->applyContactForce(&lcp->f);
    }
    sofa::helper::AdvancedTimer::stepEnd  ("Apply Contact Force");

    if(this->f_printLog.getValue())
        serr<<"applyContactForce in constraintCorrection done"<<sendl;

    sofa::helper::AdvancedTimer::stepBegin("Propagate Contact Dx");
    core::MechanicalParams mparams;
    simulation::MechanicalPropagateAndAddDxVisitor(&mparams).execute( context);
    sofa::helper::AdvancedTimer::stepEnd  ("Propagate Contact Dx");

    if(this->f_printLog.getValue())
        serr<<"propagate corrective motion done"<<sendl;

    sofa::helper::AdvancedTimer::stepBegin("Reset Contact Force");
    for (unsigned int i=0; i<constraintCorrections.size(); i++)
    {
        core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
        cc->resetContactForce();
    }
    sofa::helper::AdvancedTimer::stepEnd ("Reset Contact Force");

    if (displayTime.getValue())
    {
        sout<<" TotalTime " <<( (double) timerTotal.getTime() - timeTotal)*timeScale <<" ms" <<sendl;
    }
    return true;
}

LCP* GenericLCPConstraintSolver::getLCP()
{
    return last_lcp;
}

void GenericLCPConstraintSolver::lockLCP(LCP* l1, LCP* l2)
{
    if((lcp!=l1)&&(lcp!=l2)) // Le lcp courrant n'est pas locké
        return;

    if((&lcp1!=l1)&&(&lcp1!=l2)) // lcp1 n'est pas locké
        lcp = &lcp1;
    else if((&lcp2!=l1)&&(&lcp2!=l2)) // lcp2 n'est pas locké
        lcp = &lcp2;
    else
        lcp = &lcp3; // lcp1 et lcp2 sont lockés, donc lcp3 n'est pas locké
}


void GenericLCP::setNbConst(unsigned int nbC)
{
    dim = nbConst = nbC;
    W.resize(nbC, nbC);
    dFree.resize(nbC);
    f.resize(nbC);
    freeConstraintResolutions();
    constraintsResolutions.resize(nbC);
    _d.resize(nbC);
    _df.resize(nbC);
}

void GenericLCP::freeConstraintResolutions()
{
    for(unsigned int i=0; i<constraintsResolutions.size(); i++)
    {
        if (constraintsResolutions[i] != NULL)
        {
            delete constraintsResolutions[i];
            constraintsResolutions[i] = NULL;
        }
    }
}

// Debug is only available when called directly by the solver (not in haptic thread)
void GenericLCP::gaussSeidel(double timeout, GenericLCPConstraintSolver* solver)
{
    if(!dim)
        return;

    double t0 = (double)CTime::getTime() ;
    double timeScale = 1.0 / (double)CTime::getTicksPerSec();

    double *dfree = getDfree();
    double *force = getF();
    double **w = getW();
    double tolerance = getTolerance();

    double *d = _d.ptr();
//	double *df = _df.ptr();

    int i, j, k, l, nb;

    double errF[6];
    double error=0.0;

    bool convergence = false;
    sofa::helper::vector<double> tempForces;
    if(sor != 1.0) tempForces.resize(dim);

    if(scaleTolerance && !allVerified)
        tolerance *= dim;

    for(i=0; i<dim; )
    {
        constraintsResolutions[i]->init(i, w, force);
        i += constraintsResolutions[i]->nbLines;
    }

    sofa::helper::vector<double>* graph_residuals = NULL;
    sofa::helper::vector<double> tabErrors;

    if(solver)
    {
//		std::map < std::string, sofa::helper::vector<double> >* graph = solver->graphForces.beginEdit();
//		graph->clear();
//		solver->graphForces.endEdit();

        graph_residuals = &(*solver->graphErrors.beginEdit())["Error"];
        graph_residuals->clear();

        tabErrors.resize(dim);
    }

    /*   if(schemeCorrection)
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
    */

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
            nb = constraintsResolutions[j]->nbLines;

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
            constraintsResolutions[j]->resolution(j, w, d, force);

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

            if(constraintsResolutions[j]->tolerance)
            {
                if(contraintError > constraintsResolutions[j]->tolerance)
                    constraintsAreVerified = false;
                contraintError *= tolerance / constraintsResolutions[j]->tolerance;
            }

            error += contraintError;
            if(solver)
                tabErrors[j] = contraintError;

            j += nb;
        }

        if(solver)
        {
            /*	std::map < std::string, sofa::helper::vector<double> >* graph = solver->graphForces.beginEdit();
            	for(j=0; j<dim; j++)
            	{
            		std::ostringstream oss;
            		oss << "f" << j;

            		sofa::helper::vector<double>& graph_force = (*graph)[oss.str()];
            		graph_force.push_back(force[j]);
            	}
            	solver->graphForces.endEdit();
            */
            graph_residuals->push_back(error);
        }

        if(sor != 1.0)
        {
            for(j=0; j<dim; j++)
                force[j] = sor * force[j] + (1-sor) * tempForces[j];
        }

        double t1 = (double)CTime::getTime();
        double dt = (t1 - t0)*timeScale;

        if(timeout && dt > timeout)
            return;
        else if(allVerified)
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

    if(solver)
    {
        if(!convergence)
            solver->serr << "No convergence in gaussSeidelConstraint : error = " << error << solver->sendl;
        else if(solver->displayTime.getValue())
            solver->sout<<" Convergence after " << i+1 << " iterations " << solver->sendl;
    }

    sofa::helper::AdvancedTimer::valSet("GS iterations", i+1);

    for(i=0; i<dim; i += constraintsResolutions[i]->nbLines)
        constraintsResolutions[i]->store(i, force, convergence);
    /*
        if(schemeCorrection)
        {
            ///////// scheme correction : step 3 => the corrective motion is only based on the diff of the force value: compute this diff
            for(j=0; j<dim; j++)
            {
                df[j] += force[j];
            }
        }	*/

    if(solver)
    {
        solver->graphErrors.endEdit();

        sofa::helper::vector<double>& graph_constraints = (*solver->graphConstraints.beginEdit())["Constraints"];
        graph_constraints.clear();

        for(j=0; j<dim; )
        {
            nb = constraintsResolutions[j]->nbLines;

            if(tabErrors[j])
                graph_constraints.push_back(tabErrors[j]);
            else if(constraintsResolutions[j]->tolerance)
                graph_constraints.push_back(constraintsResolutions[j]->tolerance);
            else
                graph_constraints.push_back(tolerance);

            j += nb;
        }
        solver->graphConstraints.endEdit();
    }
}


int GenericLCPConstraintSolverClass = core::RegisterObject("A Generic Constraint Solver using the Linear Complementarity Problem formulation to solve Constraint based components")
        .add< GenericLCPConstraintSolver >();

SOFA_DECL_CLASS(GenericLCPConstraintSolver);


} // namespace constraintset

} // namespace component

} // namespace sofa
