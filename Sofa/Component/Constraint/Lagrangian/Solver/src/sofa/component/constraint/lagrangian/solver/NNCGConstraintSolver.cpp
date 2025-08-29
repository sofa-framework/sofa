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

void NNCGConstraintSolver::doSolve( SReal timeout)
{
    SCOPED_TIMER_VARNAME(unbuiltGaussSeidelTimer, "NonsmoothNonlinearConjugateGradient");


    const int dimension = current_cp->getDimension();

    if(!dimension)
    {
        current_cp->currentError = 0.0;
        current_cp->currentIterations = 0;
        return;
    }

    current_cp->m_lam.clear();
    current_cp->m_lam.resize(dimension);
    current_cp->m_deltaF.clear();
    current_cp->m_deltaF.resize(dimension);
    current_cp->m_deltaF_new.clear();
    current_cp->m_deltaF_new.resize(dimension);
    current_cp->m_p.clear();
    current_cp->m_p.resize(dimension);


    SReal *dfree = current_cp->getDfree();
    SReal *force = current_cp->getF();
    SReal **w = current_cp->getW();
    SReal tol = current_cp->tolerance;

    SReal *d = current_cp->_d.ptr();


    SReal error = 0.0;
    bool convergence = false;
    sofa::type::vector<SReal> tempForces;

    if(current_cp->sor != 1.0)
    {
        tempForces.resize(dimension);
    }

    if(current_cp->scaleTolerance && !current_cp->allVerified)
    {
        tol *= dimension;
    }

    for(int i=0; i<dimension; )
    {
        if(!current_cp->constraintsResolutions[i])
        {
            msg_error() << "Bad size of constraintsResolutions in GenericConstraintSolver" ;
            break;
        }
        current_cp->constraintsResolutions[i]->init(i, w, force);
        i += current_cp->constraintsResolutions[i]->getNbLines();
    }

    sofa::type::vector<SReal> tabErrors(dimension);

    {
        // perform one iteration of ProjectedGaussSeidel
        bool constraintsAreVerified = true;
        std::copy_n(force, dimension, std::begin(current_cp->m_lam));

        gaussSeidel_increment(false, dfree, force, w, tol, d, dimension, constraintsAreVerified, error, tabErrors);

        for(int j=0; j<dimension; j++)
        {
            current_cp->m_deltaF[j] = -(force[j] - current_cp->m_lam[j]);
            current_cp->m_p[j] = - current_cp->m_deltaF[j];
        }
    }



    int iterCount = 0;

    for(int i=1; i<d_maxIt.getValue(); i++)
    {
        iterCount ++;
        bool constraintsAreVerified = true;

        for(int j=0; j<dimension; j++)
        {
            current_cp->m_lam[j] = force[j];
        }


        error=0.0;
        gaussSeidel_increment(true, dfree, force, w, tol, d, dimension, constraintsAreVerified, error, tabErrors);


        if(current_cp->allVerified)
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
            current_cp->m_deltaF_new[j] = -(force[j] - current_cp->m_lam[j]);
        }

        const SReal beta = current_cp->m_deltaF_new.dot(current_cp->m_deltaF_new) / current_cp->m_deltaF.dot(current_cp->m_deltaF);
        current_cp->m_deltaF.eq(current_cp->m_deltaF_new, 1);

        if(beta > 1)
        {
            current_cp->m_p.clear();
            current_cp->m_p.resize(dimension);
        }
        else
        {
            for(int j=0; j<dimension; j++)
            {
                force[j] += beta*current_cp->m_p[j];
                current_cp->m_p[j] = beta*current_cp->m_p[j] -current_cp-> m_deltaF[j];
            }
        }
    }

    current_cp->result_output(this, force, error, iterCount, convergence);
}

void registerNNCGConstraintSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("A Constraint Solver using the Linear Complementarity Problem formulation to solve Constraint based components using the Non-smooth Non-linear Conjugate Gradient method")
        .add< NNCGConstraintSolver >());
}

}