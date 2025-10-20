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

#include <sofa/component/constraint/lagrangian/solver/ProjectedGaussSeidelConstraintSolver.h>
#include <sofa/component/constraint/lagrangian/solver/GenericConstraintSolver.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::constraint::lagrangian::solver
{


void ProjectedGaussSeidelConstraintSolver::doSolve( GenericConstraintProblem * problem ,SReal timeout)
{
    SCOPED_TIMER_VARNAME(gaussSeidelTimer, "ConstraintsGaussSeidel");



    const int dimension = problem->getDimension();

    if(!dimension)
    {
        problem->currentError = 0.0;
        problem->currentIterations = 0;
        return;
    }

    const SReal t0 = (SReal)sofa::helper::system::thread::CTime::getTime() ;
    const SReal timeScale = 1.0 / (SReal)sofa::helper::system::thread::CTime::getTicksPerSec();


    SReal *dfree = problem->getDfree();
    SReal *force = problem->getF();
    SReal **w = problem->getW();
    SReal tol = problem->tolerance;
    SReal *d = problem->_d.ptr();

    SReal error=0.0;
    bool convergence = false;
    sofa::type::vector<SReal> tempForces;

    if(problem->sor != 1.0)
    {
        tempForces.resize(dimension);
    }

    if(problem->scaleTolerance && !problem->allVerified)
    {
        tol *= dimension;
    }

    for(int i=0; i<dimension; )
    {
        if(!problem->constraintsResolutions[i])
        {
            msg_error()<< "Bad size of constraintsResolutions in GenericConstraintSolver" ;
            break;
        }
        problem->constraintsResolutions[i]->init(i, w, force);
        i += problem->constraintsResolutions[i]->getNbLines();
    }

    bool showGraphs = false;
    sofa::type::vector<SReal>* graph_residuals = nullptr;
    std::map < std::string, sofa::type::vector<SReal> > *graph_forces = nullptr, *graph_violations = nullptr;

    showGraphs = d_computeGraphs.getValue();

    if(showGraphs)
    {
        graph_forces = d_graphForces.beginEdit();
        graph_forces->clear();

        graph_violations = d_graphViolations.beginEdit();
        graph_violations->clear();

        graph_residuals = &(*d_graphErrors.beginEdit())["Error"];
        graph_residuals->clear();
    }

    sofa::type::vector<SReal> tabErrors(dimension);

    int iterCount = 0;


    for(int i=0; i<problem->maxIterations; i++)
    {
        iterCount ++;
        bool constraintsAreVerified = true;


        if(problem->sor != 1.0)
        {
            std::copy_n(force, dimension, tempForces.begin());
        }

        error=0.0;

        gaussSeidel_increment(true, dfree, force, w, tol, d, dimension, constraintsAreVerified, error, problem->constraintsResolutions, tabErrors);

        if(showGraphs)
        {
            for(int j=0; j<dimension; j++)
            {
                std::ostringstream oss;
                oss << "f" << j;

                sofa::type::vector<SReal>& graph_force = (*graph_forces)[oss.str()];
                graph_force.push_back(force[j]);

                sofa::type::vector<SReal>& graph_violation = (*graph_violations)[oss.str()];
                graph_violation.push_back(d[j]);
            }

            graph_residuals->push_back(error);
        }

        if(problem->sor != 1.0)
        {
            for(int j=0; j<dimension; j++)
            {
                force[j] = problem->sor * force[j] + (1-problem->sor) * tempForces[j];
            }
        }

        const SReal t1 = (SReal)sofa::helper::system::thread::CTime::getTime();
        const SReal dt = (t1 - t0)*timeScale;

        if(timeout && dt > timeout)
        {

            msg_info() <<  "TimeOut" ;

            problem->currentError = error;
            problem->currentIterations = i+1;
            return;
        }
        else if(problem->allVerified)
        {
            if(constraintsAreVerified)
            {
                convergence = true;
                break;
            }
        }
        else if(error < tol) // do not stop at the first iteration (that is used for initial guess computation)
        {
            convergence = true;
            break;
        }
    }

    sofa::helper::AdvancedTimer::valSet("GS iterations", problem->currentIterations);

    problem->result_output(this, force, error, iterCount, convergence);

    if(showGraphs)
    {
        d_graphErrors.endEdit();

        sofa::type::vector<SReal>& graph_constraints = (*d_graphConstraints.beginEdit())["Constraints"];
        graph_constraints.clear();

        for(int j=0; j<dimension; )
        {
            const unsigned int nbDofs = problem->constraintsResolutions[j]->getNbLines();

            if(tabErrors[j])
                graph_constraints.push_back(tabErrors[j]);
            else if(problem->constraintsResolutions[j]->getTolerance())
                graph_constraints.push_back(problem->constraintsResolutions[j]->getTolerance());
            else
                graph_constraints.push_back(tol);

            j += nbDofs;
        }
        d_graphConstraints.endEdit();

        d_graphForces.endEdit();
    }
}

void ProjectedGaussSeidelConstraintSolver::gaussSeidel_increment(bool measureError, SReal *dfree, SReal *force, SReal **w, SReal tol, SReal *d, int dim, bool& constraintsAreVerified, SReal& error, std::vector<core::behavior::ConstraintResolution*>& constraintCorrections, sofa::type::vector<SReal>& tabErrors) const
{
    for(int j=0; j<dim; ) // increment of j realized at the end of the loop
    {
        //1. nbLines provide the dimension of the constraint
        const unsigned int nb = constraintCorrections[j]->getNbLines();

        //2. for each line we compute the actual value of d
        //   (a)d is set to dfree

        std::vector<SReal> errF(&force[j], &force[j+nb]);
        std::copy_n(&dfree[j], nb, &d[j]);

        //   (b) contribution of forces are added to d     => TODO => optimization (no computation when force= 0 !!)
        for(int k=0; k<dim; k++)
        {
            for(unsigned int l=0; l<nb; l++)
            {
                d[j+l] += w[j+l][k] * force[k];
            }
        }

        //3. the specific resolution of the constraint(s) is called
        constraintCorrections[j]->resolution(j, w, d, force, dfree);

        //4. the error is measured (displacement due to the new resolution (i.e. due to the new force))
        if(measureError)
        {
            SReal contraintError = 0.0;
            if(nb > 1)
            {
                for(unsigned int l=0; l<nb; l++)
                {
                    SReal lineError = 0.0;
                    for (unsigned int m=0; m<nb; m++)
                    {
                        const SReal dofError = w[j+l][j+m] * (force[j+m] - errF[m]);
                        lineError += dofError * dofError;
                    }
                    lineError = sqrt(lineError);
                    if(lineError > tol)
                    {
                        constraintsAreVerified = false;
                    }

                    contraintError += lineError;
                }
            }
            else
            {
                contraintError = fabs(w[j][j] * (force[j] - errF[0]));
                if(contraintError > tol)
                {
                    constraintsAreVerified = false;
                }
            }

            const bool givenTolerance = (bool)constraintCorrections[j]->getTolerance();

            if(givenTolerance)
            {
                if(contraintError > constraintCorrections[j]->getTolerance())
                {
                    constraintsAreVerified = false;
                }
                contraintError *= tol / constraintCorrections[j]->getTolerance();
            }

            error += contraintError;
            tabErrors[j] = contraintError;
        }
        else
        {
            constraintsAreVerified = true;
        }

        j += nb;
    }
}


void registerProjectedGaussSeidelConstraintSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("A Constraint Solver using the Linear Complementarity Problem formulation to solve Constraint based components using a Projected Gauss-Seidel iterative method")
        .add< ProjectedGaussSeidelConstraintSolver >());
}


}