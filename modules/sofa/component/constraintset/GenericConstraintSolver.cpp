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

#include <sofa/component/constraintset/GenericConstraintSolver.h>

#include <sofa/simulation/common/AnimateVisitor.h>
#include <sofa/simulation/common/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/SolveVisitor.h>
#include <sofa/simulation/common/VectorOperations.h>

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

GenericConstraintSolver::GenericConstraintSolver()
    : displayTime(initData(&displayTime, false, "displayTime","Display time for each important step of GenericConstraintSolver."))
    , maxIt( initData(&maxIt, 1000, "maxIterations", "maximal number of iterations of the Gauss-Seidel algorithm"))
    , tolerance( initData(&tolerance, 0.001, "tolerance", "residual error threshold for termination of the Gauss-Seidel algorithm"))
    , sor( initData(&sor, 1.0, "sor", "Successive Over Relaxation parameter (0-2)"))
    , scaleTolerance( initData(&scaleTolerance, true, "scaleTolerance", "Scale the error tolerance with the number of constraints"))
    , allVerified( initData(&allVerified, false, "allVerified", "All contraints must be verified (each constraint's error < tolerance)"))
    , schemeCorrection( initData(&schemeCorrection, false, "schemeCorrection", "Apply new scheme where compliance is progressively corrected"))
    , unbuilt(initData(&unbuilt, false, "unbuilt", "Compliance is not fully built"))
    , graphErrors( initData(&graphErrors,"graphErrors","Sum of the constraints' errors at each iteration"))
    , graphConstraints( initData(&graphConstraints,"graphConstraints","Graph of each constraint's error at the end of the resolution"))
//, graphForces( initData(&graphForces,"graphForces","Graph of each constraint's force at each step of the resolution"))
    , current_cp(&cp1)
    , last_cp(NULL)
{
    addAlias(&maxIt, "maxIt");

    graphErrors.setWidget("graph");
    graphErrors.setGroup("Graph");

    graphConstraints.setWidget("graph");
    graphConstraints.setGroup("Graph");

//	graphForces.setWidget("graph");
//	graphForces.setGroup("Graph2");
}

GenericConstraintSolver::~GenericConstraintSolver()
{
}

void GenericConstraintSolver::init()
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

bool GenericConstraintSolver::prepareStates(const core::ConstraintParams *cParams, MultiVecId /*res1*/, MultiVecId /*res2*/)
{
    sofa::helper::AdvancedTimer::StepVar vtimer("PrepareStates");

    last_cp = current_cp;

    time = 0.0;
    timeTotal = 0.0;
    timeScale = 1000.0 / (double)CTime::getTicksPerSec();

    simulation::common::VectorOperations vop(cParams, this->getContext());
    vop.v_clear(this->m_fId);
    vop.v_clear(this->m_dxId);

    if ( displayTime.getValue() )
    {
        time = (double) timer.getTime();
        timeTotal = (double) timerTotal.getTime();
    }

    return true;
}


void afficheLCP(double *q, double **M, double *f, int dim)
{
    int compteur, compteur2;
    // affichage de la matrice du LCP
    printf("\n M = [");
    for(compteur=0; compteur<dim; compteur++)
    {
        for(compteur2=0; compteur2<dim; compteur2++)
        {
            printf("\t%.9f",M[compteur][compteur2]);
        }
        printf("\n");
    }
    printf("      ];\n\n");

    // affichage de q
    printf("q = [");
    for(compteur=0; compteur<dim; compteur++)
    {
        printf("\t%.9f\n",q[compteur]);
    }
    printf("      ];\n\n");

    // affichage de f
    printf("f = [");
    for(compteur=0; compteur<dim; compteur++)
    {
        printf("\t%.9f\n",f[compteur]);
    }
    printf("      ];\n\n");
}


bool GenericConstraintSolver::buildSystem(const core::ConstraintParams *cParams, MultiVecId /*res1*/, MultiVecId /*res2*/)
{
    unsigned int numConstraints = 0;

    sofa::helper::AdvancedTimer::stepBegin("Accumulate Constraint");
    // mechanical action executed from root node to propagate the constraints
    simulation::MechanicalResetConstraintVisitor(cParams).execute(context);
    // calling buildConstraintMatrix
    //simulation::MechanicalAccumulateConstraint(&cparams /* PARAMS FIRST */, core::MatrixDerivId::holonomicC(), numConstraints).execute(context);

    MechanicalSetConstraint(cParams, core::MatrixDerivId::holonomicC(), numConstraints).execute(context);
    MechanicalAccumulateConstraint2(cParams, core::MatrixDerivId::holonomicC()).execute(context);

    sofa::helper::AdvancedTimer::stepEnd  ("Accumulate Constraint");
    sofa::helper::AdvancedTimer::valSet("numConstraints", numConstraints);

    current_cp->clear(numConstraints);

    sofa::helper::AdvancedTimer::stepBegin("Get Constraint Value");
    MechanicalGetConstraintViolationVisitor(cParams /* PARAMS FIRST */, &current_cp->dFree).execute(context);
    sofa::helper::AdvancedTimer::stepEnd ("Get Constraint Value");

    sofa::helper::AdvancedTimer::stepBegin("Get Constraint Resolutions");
    MechanicalGetConstraintResolutionVisitor(cParams /* PARAMS FIRST */, current_cp->constraintsResolutions).execute(context);
    sofa::helper::AdvancedTimer::stepEnd("Get Constraint Resolutions");

    if (this->f_printLog.getValue()) sout<<"GenericConstraintSolver: "<<numConstraints<<" constraints"<<sendl;

    if(unbuilt.getValue())
    {
        for (unsigned int i=0; i<constraintCorrections.size(); i++)
        {
            core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
            cc->resetForUnbuiltResolution(current_cp->getF(), current_cp->constraints_sequence);
        }

        SparseMatrix<double>* Wdiag = &current_cp->Wdiag;
        Wdiag->resize(numConstraints, numConstraints);

        // for each contact, the pair of constraint correction that is involved with the contact is memorized
        current_cp->cclist_elem1.resize(numConstraints);
        current_cp->cclist_elem2.resize(numConstraints);
        unsigned int nbObjects = 0;
        for(unsigned int i=0; i<numConstraints;)
        {
            nbObjects++;
            unsigned int l = current_cp->constraintsResolutions[i]->nbLines;

            bool elem1 = false, elem2 = false;
            for (unsigned int j=0; j<constraintCorrections.size(); j++)
            {
                core::behavior::BaseConstraintCorrection* cc = constraintCorrections[j];
                if(cc->hasConstraintNumber(i))
                {
                    if(elem1)
                    {
                        current_cp->cclist_elem2[i] = cc;
                        elem2=true;
                    }
                    else
                    {
                        current_cp->cclist_elem1[i] = cc;
                        elem1=true;
                    }

                }
            }

            if(elem1)	// for each contact, the pair of constraintcorrection is called to add the contribution
                current_cp->cclist_elem1[i]->getBlockDiagonalCompliance(Wdiag, i, i+l-1);
            else
                serr<<"WARNING: no constraintCorrection found for constraint"<<i<<sendl;

            if(elem2)
                current_cp->cclist_elem2[i]->getBlockDiagonalCompliance(Wdiag, i, i+l-1);
            else
                current_cp->cclist_elem2[i] = (NULL);

            double** w =  current_cp->getW();
            for(unsigned int m=i; m<i+l; m++)
                for(unsigned int n=i; n<i+l; n++)
                    w[m][n] = Wdiag->element(m, n);

            i += l;
        }

        current_cp->change_sequence = false;
        if(current_cp->constraints_sequence.size() == nbObjects)
            current_cp->change_sequence=true;
    }
    else
    {
        sofa::helper::AdvancedTimer::stepBegin("Get Compliance");
        if (this->f_printLog.getValue()) sout<<" computeCompliance in "  << constraintCorrections.size()<< " constraintCorrections" <<sendl;
        for (unsigned int i=0; i<constraintCorrections.size(); i++)
        {
            core::behavior::BaseConstraintCorrection* cc = constraintCorrections[i];
            cc->addComplianceInConstraintSpace(cParams, &current_cp->W);
        }
        sofa::helper::AdvancedTimer::stepEnd  ("Get Compliance");
        if (this->f_printLog.getValue()) sout<<" computeCompliance_done "  <<sendl;
    }


    if ( displayTime.getValue() )
    {
        sout<<" build_LCP " << ( (double) timer.getTime() - time)*timeScale<<" ms" <<sendl;
        time = (double) timer.getTime();
    }
    return true;
}



bool GenericConstraintSolver::solveSystem(const core::ConstraintParams * /*cParams*/, MultiVecId /*res1*/, MultiVecId /*res2*/)
{
    current_cp->tolerance = tolerance.getValue();
    current_cp->maxIterations = maxIt.getValue();
    current_cp->scaleTolerance = scaleTolerance.getValue();
    current_cp->allVerified = allVerified.getValue();
    current_cp->sor = sor.getValue();

    if(unbuilt.getValue())
    {
        sofa::helper::AdvancedTimer::stepBegin("ConstraintsUnbuiltGaussSeidel");
        current_cp->unbuiltGaussSeidel(0, this);
        sofa::helper::AdvancedTimer::stepEnd("ConstraintsUnbuiltGaussSeidel");
    }
    else
    {
        sofa::helper::AdvancedTimer::stepBegin("ConstraintsGaussSeidel");
        current_cp->gaussSeidel(0, this);
        sofa::helper::AdvancedTimer::stepEnd("ConstraintsGaussSeidel");
    }

    if ( displayTime.getValue() )
    {
        sout<<" TOTAL solve_LCP " <<( (double) timer.getTime() - time)*timeScale<<" ms" <<sendl;
        time = (double) timer.getTime();
    }

    if(this->f_printLog.getValue())
        afficheLCP(current_cp->getDfree(), current_cp->getW(), current_cp->getF(), current_cp->getDimension());

    return true;
}

bool GenericConstraintSolver::applyCorrection(const core::ConstraintParams *cParams, MultiVecId res1, MultiVecId res2)
{
    using sofa::helper::AdvancedTimer;
    using core::behavior::BaseConstraintCorrection;

    if (this->f_printLog.getValue())
        serr << "KeepContactForces done" << sendl;

    AdvancedTimer::stepBegin("Compute And Apply Motion Correction");

    if (cParams->constOrder() == core::ConstraintParams::POS_AND_VEL)
    {
        core::MultiVecCoordId xId(res1);
        core::MultiVecDerivId vId(res2);
        for (unsigned int i = 0; i < constraintCorrections.size(); i++)
        {
            BaseConstraintCorrection* cc = constraintCorrections[i];
            cc->computeAndApplyMotionCorrection(cParams, xId, vId, this->m_fId, &current_cp->f);
        }
    }
    else if (cParams->constOrder() == core::ConstraintParams::POS)
    {
        core::MultiVecCoordId xId(res1);
        for (unsigned int i = 0; i < constraintCorrections.size(); i++)
        {
            BaseConstraintCorrection* cc = constraintCorrections[i];
            cc->computeAndApplyPositionCorrection(cParams, xId, this->m_fId, &current_cp->f);
        }
    }
    else if (cParams->constOrder() == core::ConstraintParams::VEL)
    {
        core::MultiVecDerivId vId(res1);
        for (unsigned int i = 0; i < constraintCorrections.size(); i++)
        {
            BaseConstraintCorrection* cc = constraintCorrections[i];
            cc->computeAndApplyVelocityCorrection(cParams, vId, this->m_fId, &current_cp->f);
        }
    }

    AdvancedTimer::stepEnd("Compute And Apply Motion Correction");

    if (this->f_printLog.getValue())
        serr << "Compute And Apply Motion Correction in constraintCorrection done" << sendl;

    if (displayTime.getValue())
    {
        sout << " TotalTime " << ((double) timerTotal.getTime() - timeTotal) * timeScale << " ms" << sendl;
    }

    return true;
}


ConstraintProblem* GenericConstraintSolver::getConstraintProblem()
{
    return last_cp;
}

void GenericConstraintSolver::lockConstraintProblem(ConstraintProblem* p1, ConstraintProblem* p2)
{
    if( (current_cp != p1) && (current_cp != p2) ) // Le ConstraintProblem courant n'est pas locké
        return;

    if( (&cp1 != p1) && (&cp1 != p2) ) // cp1 n'est pas locké
        current_cp = &cp1;
    else if( (&cp2 != p1) && (&cp2 != p2) ) // cp2 n'est pas locké
        current_cp = &cp2;
    else
        current_cp = &cp3; // cp1 et cp2 sont lockés, donc cp3 n'est pas locké
}

void GenericConstraintProblem::clear(int nbC)
{
    ConstraintProblem::clear(nbC);

    freeConstraintResolutions();
    constraintsResolutions.resize(nbC);
    _d.resize(nbC);
}

void GenericConstraintProblem::freeConstraintResolutions()
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

void GenericConstraintProblem::solveTimed(double tol, int maxIt, double timeout)
{
    double tempTol = tolerance;
    int tempMaxIt = maxIterations;

    tolerance = tol;
    maxIterations = maxIt;

    gaussSeidel(timeout);

    tolerance = tempTol;
    maxIterations = tempMaxIt;
}

// Debug is only available when called directly by the solver (not in haptic thread)
void GenericConstraintProblem::gaussSeidel(double timeout, GenericConstraintSolver* solver)
{
    if(!dimension)
        return;

    double t0 = (double)CTime::getTime() ;
    double timeScale = 1.0 / (double)CTime::getTicksPerSec();

    double *dfree = getDfree();
    double *force = getF();
    double **w = getW();
    double tol = tolerance;

    double *d = _d.ptr();

    int i, j, k, l, nb;

    double errF[6];
    double error=0.0;

    bool convergence = false;
    sofa::helper::vector<double> tempForces;
    if(sor != 1.0) tempForces.resize(dimension);

    if(scaleTolerance && !allVerified)
        tol *= dimension;

    for(i=0; i<dimension; )
    {
        if(!constraintsResolutions[i])
        {
            std::cerr << "Bad size of constraintsResolutions in GenericConstraintProblem" << std::endl;
            dimension = i;
            break;
        }
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

        tabErrors.resize(dimension);
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

    for(i=0; i<maxIterations; i++)
    {
        bool constraintsAreVerified = true;
        if(sor != 1.0)
        {
            for(j=0; j<dimension; j++)
                tempForces[j] = force[j];
        }

        error=0.0;
        for(j=0; j<dimension; ) // increment of j realized at the end of the loop
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
            //   (b) contribution of forces are added to d     => TODO => optimization (no computation when force= 0 !!)
            for(k=0; k<dimension; k++)
                for(l=0; l<nb; l++)
                    d[j+l] += w[j+l][k] * force[k];

            //3. the specific resolution of the constraint(s) is called
            constraintsResolutions[j]->resolution(j, w, d, force, dfree);

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
                    if(lineError > tol)
                        constraintsAreVerified = false;

                    contraintError += lineError;
                }
            }
            else
            {
                contraintError = fabs(w[j][j] * (force[j] - errF[0]));
                if(contraintError > tol)
                    constraintsAreVerified = false;
            }

            if(constraintsResolutions[j]->tolerance)
            {
                if(contraintError > constraintsResolutions[j]->tolerance)
                    constraintsAreVerified = false;
                contraintError *= tol / constraintsResolutions[j]->tolerance;
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
            for(j=0; j<dimension; j++)
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
        else if(error < tol/* && i>0*/) // do not stop at the first iteration (that is used for initial guess computation)
        {
            convergence = true;
            break;
        }
    }

    if(solver)
    {
        if(!convergence)
        {
            if(solver->f_printLog.getValue())
                solver->serr << "No convergence : error = " << error << solver->sendl;
            else
                solver->sout << "No convergence : error = " << error << solver->sendl;
        }
        else if(solver->displayTime.getValue())
            solver->sout<<" Convergence after " << i+1 << " iterations " << solver->sendl;
    }

    sofa::helper::AdvancedTimer::valSet("GS iterations", i+1);

    for(i=0; i<dimension; i += constraintsResolutions[i]->nbLines)
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

        for(j=0; j<dimension; )
        {
            nb = constraintsResolutions[j]->nbLines;

            if(tabErrors[j])
                graph_constraints.push_back(tabErrors[j]);
            else if(constraintsResolutions[j]->tolerance)
                graph_constraints.push_back(constraintsResolutions[j]->tolerance);
            else
                graph_constraints.push_back(tol);

            j += nb;
        }
        solver->graphConstraints.endEdit();
    }
}


void GenericConstraintProblem::unbuiltGaussSeidel(double timeout, GenericConstraintSolver* solver)
{
    if(!dimension)
        return;

    double t0 = (double)CTime::getTime() ;
    double timeScale = 1.0 / (double)CTime::getTicksPerSec();

    double *dfree = getDfree();
    double *force = getF();
    double **w = getW();
    double tol = tolerance;

    double *d = _d.ptr();

    int i, j, l, nb;

    double errF[6];
    double error=0.0;

    bool convergence = false;
    sofa::helper::vector<double> tempForces;
    if(sor != 1.0) tempForces.resize(dimension);

    if(scaleTolerance && !allVerified)
        tol *= dimension;

    for(i=0; i<dimension; )
    {
        constraintsResolutions[i]->init(i, w, force);
        int nb = constraintsResolutions[i]->nbLines;
        // TODO : previous forces don't work with multiple constraints
//		if(cclist_elem1[i]) cclist_elem1[i]->setConstraintDForce(force, i, i+nb-1, true);
//		if(cclist_elem2[i]) cclist_elem2[i]->setConstraintDForce(force, i, i+nb-1, true);
        i += nb;
    }
    memset(force, 0, dimension * sizeof(double));	// Erase previous forces for the time being

    sofa::helper::vector<double>* graph_residuals = NULL;
    sofa::helper::vector<double> tabErrors;

    if(solver)
    {
        graph_residuals = &(*solver->graphErrors.beginEdit())["Error"];
        graph_residuals->clear();

        tabErrors.resize(dimension);
    }

    for(i=0; i<maxIterations; i++)
    {
        bool constraintsAreVerified = true;
        if(sor != 1.0)
        {
            for(j=0; j<dimension; j++)
                tempForces[j] = force[j];
        }

        error=0.0;
        for(j=0; j<dimension; ) // increment of j realized at the end of the loop
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
            if(cclist_elem1[j]) cclist_elem1[j]->addConstraintDisplacement(d, j, j+nb-1);
            if(cclist_elem2[j]) cclist_elem2[j]->addConstraintDisplacement(d, j, j+nb-1);

            //3. the specific resolution of the constraint(s) is called
            constraintsResolutions[j]->resolution(j, w, d, force, dfree);

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
                    if(lineError > tol)
                        constraintsAreVerified = false;

                    contraintError += lineError;
                }
            }
            else
            {
                contraintError = fabs(w[j][j] * (force[j] - errF[0]));
                if(contraintError > tol)
                    constraintsAreVerified = false;
            }

            if(constraintsResolutions[j]->tolerance)
            {
                if(contraintError > constraintsResolutions[j]->tolerance)
                    constraintsAreVerified = false;
                contraintError *= tol / constraintsResolutions[j]->tolerance;
            }

            error += contraintError;
            if(solver)
                tabErrors[j] = contraintError;

            //5. the force is updated for the constraint corrections
            bool update = false;
            for(l=0; l<nb; l++)
                update |= (force[j+l] || errF[l]);

            if(update)
            {
                double tempF[6];
                for(l=0; l<nb; l++)
                {
                    tempF[l] = force[j+l];
                    force[j+l] -= errF[l]; // DForce
                }

                if(cclist_elem1[j]) cclist_elem1[j]->setConstraintDForce(force, j, j+nb-1, update);
                if(cclist_elem2[j]) cclist_elem2[j]->setConstraintDForce(force, j, j+nb-1, update);

                for(l=0; l<nb; l++)
                    force[j+l] = tempF[l];
            }

            j += nb;
        }

        if(solver)
            graph_residuals->push_back(error);

        if(sor != 1.0)
        {
            for(j=0; j<dimension; j++)
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
        else if(error < tol/* && i>0*/) // do not stop at the first iteration (that is used for initial guess computation)
        {
            convergence = true;
            break;
        }
    }

    if(solver)
    {
        if(!convergence)
        {
            if(solver->f_printLog.getValue())
                solver->serr << "No convergence : error = " << error << solver->sendl;
            else
                solver->sout << "No convergence : error = " << error << solver->sendl;
        }
        else if(solver->displayTime.getValue())
            solver->sout<<" Convergence after " << i+1 << " iterations " << solver->sendl;
    }

    sofa::helper::AdvancedTimer::valSet("GS iterations", i+1);

    for(i=0; i<dimension; i += constraintsResolutions[i]->nbLines)
        constraintsResolutions[i]->store(i, force, convergence);

    if(solver)
    {
        solver->graphErrors.endEdit();

        sofa::helper::vector<double>& graph_constraints = (*solver->graphConstraints.beginEdit())["Constraints"];
        graph_constraints.clear();

        for(j=0; j<dimension; )
        {
            nb = constraintsResolutions[j]->nbLines;

            if(tabErrors[j])
                graph_constraints.push_back(tabErrors[j]);
            else if(constraintsResolutions[j]->tolerance)
                graph_constraints.push_back(constraintsResolutions[j]->tolerance);
            else
                graph_constraints.push_back(tol);

            j += nb;
        }
        solver->graphConstraints.endEdit();
    }
}



int GenericConstraintSolverClass = core::RegisterObject("A Generic Constraint Solver using the Linear Complementarity Problem formulation to solve Constraint based components")
        .add< GenericConstraintSolver >();

SOFA_DECL_CLASS(GenericConstraintSolver);


} // namespace constraintset

} // namespace component

} // namespace sofa
