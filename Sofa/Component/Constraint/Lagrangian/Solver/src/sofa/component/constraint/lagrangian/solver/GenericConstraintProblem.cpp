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
#include <sofa/component/constraint/lagrangian/solver/GenericConstraintProblem.h>
#include <sofa/core/behavior/ConstraintResolution.h>

#include <sofa/component/constraint/lagrangian/solver/GenericConstraintSolver.h>
#include <sofa/helper/AdvancedTimer.h>

namespace sofa::component::constraint::lagrangian::solver
{

void GenericConstraintProblem::clear(int nbC)
{
    ConstraintProblem::clear(nbC);

    freeConstraintResolutions();
    constraintsResolutions.resize(nbC);
    _d.resize(nbC);
}

void GenericConstraintProblem::freeConstraintResolutions()
{
    for(auto*& constraintsResolution : constraintsResolutions)
    {
        delete constraintsResolution;
        constraintsResolution = nullptr;
    }
}

int GenericConstraintProblem::getNumConstraints()
{
    return dimension;
}

int GenericConstraintProblem::getNumConstraintGroups()
{
    int n = 0;
    for(int i=0; i<dimension; )
    {
        if(!constraintsResolutions[i])
        {
            break;
        }
        ++n;
        i += constraintsResolutions[i]->getNbLines();
    }
    return n;
}

void GenericConstraintProblem::solveTimed(SReal tol, int maxIt, SReal timeout)
{
    const SReal tempTol = tolerance;
    const int tempMaxIt = maxIterations;

    tolerance = tol;
    maxIterations = maxIt;


    m_solver->doSolve(this, timeout);

    tolerance = tempTol;
    maxIterations = tempMaxIt;
}

void GenericConstraintProblem::setSolver(GenericConstraintSolver* solver)
{
    m_solver = solver;
}

void GenericConstraintProblem::result_output(GenericConstraintSolver *solver, SReal *force, SReal error, int iterCount, bool convergence)
{
    currentError = error;
    currentIterations = iterCount+1;

    sofa::helper::AdvancedTimer::valSet("GS iterations", currentIterations);

    if(!convergence)
    {
        msg_info(solver) << "No convergence : error = " << error ;
    }
    else
    {
        msg_info(solver) << "Convergence after " << currentIterations << " iterations " ;
    }

    for(int i=0; i<dimension; i += constraintsResolutions[i]->getNbLines())
    {
        constraintsResolutions[i]->store(i, force, convergence);
    }
}




}
