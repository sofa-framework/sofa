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

#include <sofa/component/constraint/lagrangian/solver/NNCGConstraintProblem.h>
#include <sofa/component/constraint/lagrangian/solver/GenericConstraintSolver.h>
#include <sofa/helper/ScopedAdvancedTimer.h>


namespace sofa::component::constraint::lagrangian::solver
{

void NNCGConstraintProblem::solve( SReal timeout, GenericConstraintSolver* solver)
{
    SCOPED_TIMER_VARNAME(unbuiltGaussSeidelTimer, "NonsmoothNonlinearConjugateGradient");

    if(!solver)
        return;

    const int dimension = getDimension();

    if(!dimension)
    {
        currentError = 0.0;
        currentIterations = 0;
        return;
    }

    m_lam.clear();
    m_lam.resize(dimension);
    m_deltaF.clear();
    m_deltaF.resize(dimension);
    m_deltaF_new.clear();
    m_deltaF_new.resize(dimension);
    m_p.clear();
    m_p.resize(dimension);


    SReal *dfree = getDfree();
    SReal *force = getF();
    SReal **w = getW();
    SReal tol = tolerance;

    SReal *d = _d.ptr();


    SReal error = 0.0;
    bool convergence = false;
    sofa::type::vector<SReal> tempForces;

    if(sor != 1.0)
    {
        tempForces.resize(dimension);
    }

    if(scaleTolerance && !allVerified)
    {
        tol *= dimension;
    }

    for(int i=0; i<dimension; )
    {
        if(!constraintsResolutions[i])
        {
            msg_error(solver) << "Bad size of constraintsResolutions in GenericConstraintProblem" ;
            break;
        }
        constraintsResolutions[i]->init(i, w, force);
        i += constraintsResolutions[i]->getNbLines();
    }

    sofa::type::vector<SReal> tabErrors(dimension);

    {
        // perform one iteration of ProjectedGaussSeidel
        bool constraintsAreVerified = true;
        std::copy_n(force, dimension, std::begin(m_lam));

        gaussSeidel_increment(false, dfree, force, w, tol, d, dimension, constraintsAreVerified, error, tabErrors);

        for(int j=0; j<dimension; j++)
        {
            m_deltaF[j] = -(force[j] - m_lam[j]);
            m_p[j] = - m_deltaF[j];
        }
    }



    int iterCount = 0;

    for(int i=1; i<solver->d_maxIt.getValue(); i++)
    {
        iterCount ++;
        bool constraintsAreVerified = true;

        for(int j=0; j<dimension; j++)
        {
            m_lam[j] = force[j];
        }


        error=0.0;
        gaussSeidel_increment(true, dfree, force, w, tol, d, dimension, constraintsAreVerified, error, tabErrors);


        if(allVerified)
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
            m_deltaF_new[j] = -(force[j] - m_lam[j]);
        }

        const SReal beta = m_deltaF_new.dot(m_deltaF_new) / m_deltaF.dot(m_deltaF);
        m_deltaF.eq(m_deltaF_new, 1);

        if(beta > 1)
        {
            m_p.clear();
            m_p.resize(dimension);
        }
        else
        {
            for(int j=0; j<dimension; j++)
            {
                force[j] += beta*m_p[j];
                m_p[j] = beta*m_p[j] - m_deltaF[j];
            }
        }
    }

    result_output(solver, force, error, iterCount, convergence);
}

}