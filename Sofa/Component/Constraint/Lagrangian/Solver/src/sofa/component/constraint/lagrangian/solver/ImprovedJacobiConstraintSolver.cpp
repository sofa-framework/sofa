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

#include <sofa/component/constraint/lagrangian/solver/ImprovedJacobiConstraintSolver.h>
#include <sofa/component/constraint/lagrangian/solver/GenericConstraintSolver.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/core/ObjectFactory.h>
#include <Eigen/Eigenvalues>

namespace sofa::component::constraint::lagrangian::solver
{

void ImprovedJacobiConstraintSolver::doSolve( SReal timeout)
{
    SCOPED_TIMER_VARNAME(gaussSeidelTimer, "ConstraintsGaussSeidel");


    const int dimension = current_cp->getDimension();

    if(!dimension)
    {
        current_cp->currentError = 0.0;
        current_cp->currentIterations = 0;
        return;
    }

    SReal *dfree = current_cp->getDfree();
    SReal *force = current_cp->getF();
    SReal **w = current_cp->getW();
    SReal tol = current_cp->tolerance;
    SReal *d = current_cp->_d.ptr();

    std::copy_n(dfree, dimension, d);

    for(unsigned i=0; i< dimension; ++i)
    {
        force[i] = 0;
    }

    std::vector<SReal> lastF;
    lastF.resize(current_cp->getDimension(), 0.0);

    std::vector<SReal> deltaF;
    deltaF.resize(current_cp->getDimension(), 0.0);

    std::vector<SReal> correctedD;
    correctedD.resize(current_cp->getDimension(), 0.0);

//    std::cout<<"Initialized vectors"<<std::endl;

    SReal error=0.0;
    bool convergence = false;
    if(current_cp->scaleTolerance && !current_cp->allVerified)
    {
        tol *= dimension;
    }

    for(int i=0; i<dimension; )
    {
        if(!current_cp->constraintsResolutions[i])
        {
            msg_error()<< "Bad size of constraintsResolutions in GenericConstraintSolver" ;
            break;
        }
        current_cp->constraintsResolutions[i]->init(i, w, force);
        i += current_cp->constraintsResolutions[i]->getNbLines();
    }

    sofa::type::vector<SReal> tabErrors(dimension);

    int iterCount = 0;

    Eigen::Map<Eigen::MatrixX<SReal>> EigenW(w[0],dimension, dimension) ;
    SReal eigenRadius = 0;
    for(auto s : EigenW.eigenvalues())
    {
        eigenRadius=std::max(eigenRadius,norm(s));
    }
    const SReal rho = std::min(1.0, 0.9 * 2/eigenRadius);

    for(int i=0; i<current_cp->maxIterations; i++)
    {
        iterCount ++;
        bool constraintsAreVerified = true;

        error=0.0;
        SReal beta = std::min(1.0, pow( ((float)i)/current_cp->maxIterations,0.6));

        for(int j=0; j<dimension; ) // increment of j realized at the end of the loop
        {
            // 1. nbLines provide the dimension of the constraint
            const unsigned int nb = current_cp->constraintsResolutions[j]->getNbLines();

            for(unsigned l=j; l<j+nb; ++l )
            {
                for(unsigned k=0; k<dimension; ++k)
                {
                    d[l] +=  w[l][k] * deltaF[k];
                }
                correctedD[l] = rho * d[l]  ;
            }
            current_cp->constraintsResolutions[j]->resolution(j,w,correctedD.data(), force, dfree);
            for(unsigned l=j; l<j+nb; ++l )
            {
                force[l] += beta * deltaF[l] ;
                deltaF[l] = force[l] - lastF[l];
                lastF[l] = force[l];
            }

            double cstError = 0.0;
            for(unsigned l=j; l<j+nb; ++l )
            {
                for(unsigned k=0; k<dimension; ++k)
                {
                    cstError += pow(w[l][k] * deltaF[k],2);
                }
                constraintsAreVerified = constraintsAreVerified && cstError < pow(tol,2);
            }
            error += sqrt(cstError);
            j+= nb;

        }

        if (current_cp->allVerified && constraintsAreVerified)
        {
            convergence = true;
            return;
        }

        if(error < tol && i > 0) // do not stop at the first iteration (that is used for initial guess computation)
        {
            convergence = true;
            break;
        }
    }

    sofa::helper::AdvancedTimer::valSet("GS iterations", current_cp->currentIterations);

    current_cp->result_output(this, force, error, iterCount, convergence);

}



void registerImprovedJacobiConstraintSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("A Constraint Solver using the Linear Complementarity Problem formulation to solve Constraint based components using a Projected Gauss-Seidel iterative method")
        .add< ImprovedJacobiConstraintSolver >());
}


}