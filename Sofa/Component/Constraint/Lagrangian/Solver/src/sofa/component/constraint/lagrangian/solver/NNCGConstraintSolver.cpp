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

#include <sofa/component/constraint/lagrangian/solver/NNCGConstraintSolver.h>
#include <sofa/component/constraint/lagrangian/solver/GenericConstraintSolver.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/core/ObjectFactory.h>


namespace sofa::component::constraint::lagrangian::solver
{

void NNCGConstraintSolver::doSolve(GenericConstraintProblem * problem , SReal timeout)
{
    SCOPED_TIMER_VARNAME(unbuiltGaussSeidelTimer, "NonsmoothNonlinearConjugateGradient");

    const int dimension = problem->getDimension();

    if(!dimension)
    {
        problem->currentError = 0.0;
        problem->currentIterations = 0;
        return;
    }

    problem->m_lam.clear();
    problem->m_lam.resize(dimension);
    problem->m_deltaF.clear();
    problem->m_deltaF.resize(dimension);
    problem->m_deltaF_new.clear();
    problem->m_deltaF_new.resize(dimension);
    problem->m_p.clear();
    problem->m_p.resize(dimension);


    SReal *dfree = problem->getDfree();
    SReal *force = problem->getF();
    SReal **w = problem->getW();
    SReal tol = problem->tolerance;

    SReal *d = problem->_d.ptr();


    SReal error = 0.0;
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
            msg_error() << "Bad size of constraintsResolutions in GenericConstraintSolver" ;
            break;
        }

        problem->constraintsResolutions[i]->init(i, w, force);
        i += problem->constraintsResolutions[i]->getNbLines();
    }

    sofa::type::vector<SReal> tabErrors(dimension);

    {
        // perform one iteration of ProjectedGaussSeidel
        bool constraintsAreVerified = true;

        std::copy_n(force, dimension, std::begin(problem->m_lam));

        gaussSeidel_increment(false, dfree, force, w, tol, d, dimension, constraintsAreVerified, error, problem->constraintsResolutions,  tabErrors);

        for(int j=0; j<dimension; j++)
        {
            problem->m_deltaF[j] = -(force[j] - problem->m_lam[j]);
            problem->m_p[j] = - problem->m_deltaF[j];
        }
    }



    int iterCount = 0;

    for(int i=1; i<d_maxIt.getValue(); i++)
    {
        iterCount ++;
        bool constraintsAreVerified = true;

        for(int j=0; j<dimension; j++)
        {
            problem->m_lam[j] = force[j];
        }


        error=0.0;

        gaussSeidel_increment(true, dfree, force, w, tol, d, dimension, constraintsAreVerified, error, problem->constraintsResolutions, tabErrors);


        if(problem->allVerified)
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


        // NNCG update with the correction p
        for(int j=0; j<dimension; j++)
        {
            problem->m_deltaF_new[j] = -(force[j] - problem->m_lam[j]);
        }

        const SReal beta = problem->m_deltaF_new.dot(problem->m_deltaF_new) / problem->m_deltaF.dot(problem->m_deltaF);
        problem->m_deltaF.eq(problem->m_deltaF_new, 1);

        if(beta > 1)
        {
            problem->m_p.clear();
            problem->m_p.resize(dimension);
        }
        else
        {
            for(int j=0; j<dimension; j++)
            {
                force[j] += beta*problem->m_p[j];
                problem->m_p[j] = beta*problem->m_p[j] -problem-> m_deltaF[j];
            }
        }
        //Stopping condition based on constraint evolution rate
        if (problem->m_deltaF_new.norm() < tol)
        {
            break;
        }
    }

    problem->result_output(this, force, error, iterCount, convergence);
}

void registerNNCGConstraintSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("A Constraint Solver using the Linear Complementarity Problem formulation to solve Constraint based components using the Non-smooth Non-linear Conjugate Gradient method")
        .add< NNCGConstraintSolver >());
}

}