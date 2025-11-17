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

ImprovedJacobiConstraintSolver::ImprovedJacobiConstraintSolver()
    : BuiltConstraintSolver()
    , d_useSpectralCorrection(initData(&d_useSpectralCorrection,false,"useSpectralCorrection","If set to true, the solution found after each iteration will be multiplied by spectralCorrectionFactor*2/spr(W), with spr() denoting the spectral radius."))
    , d_spectralCorrectionFactor(initData(&d_spectralCorrectionFactor,1.0,"spectralCorrectionFactor","Factor used to modulate the spectral correction"))
    , d_useConjugateResidue(initData(&d_useConjugateResidue,false,"useConjugateResidue","If set to true, the solution found after each iteration will be corrected along the solution direction using `\\lambda^{i+1} -= beta^{i} * (\\lambda^{i} - \\lambda^{i-1})` with beta following the formula beta^{i} = min(1, (i/maxIterations)^{conjugateResidueSpeedFactor}) "))
    , d_conjugateResidueSpeedFactor(initData(&d_conjugateResidueSpeedFactor,10.0,"conjugateResidueSpeedFactor","Factor used to modulate the speed in which beta used in the conjugate residue part reaches 1.0. The higher the value, the slower the reach. "))
{

}


void ImprovedJacobiConstraintSolver::doSolve(GenericConstraintProblem * problem , SReal timeout)
{
    SCOPED_TIMER_VARNAME(gaussSeidelTimer, "ImprovedJacobiConstraintSolver");


    const int dimension = problem->getDimension();

    if(!dimension)
    {
        problem->currentError = 0.0;
        problem->currentIterations = 0;
        return;
    }

    SReal *dfree = problem->getDfree();
    SReal *force = problem->getF();
    SReal **w = problem->getW();
    SReal tol = problem->tolerance;
    SReal *d = problem->_d.ptr();

    std::copy_n(dfree, dimension, d);

    for(unsigned i=0; i< dimension; ++i)
    {
        force[i] = 0;
    }

    std::vector<SReal> lastF;
    lastF.resize(problem->getDimension(), 0.0);

    std::vector<SReal> deltaF;
    deltaF.resize(problem->getDimension(), 0.0);

    std::vector<SReal> correctedD;
    correctedD.resize(problem->getDimension(), 0.0);


    SReal error=0.0;
    bool convergence = false;
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

    sofa::type::vector<SReal> tabErrors(dimension);

    int iterCount = 0;

    SReal rho = 1.0;

    if (d_useSpectralCorrection.getValue())
    {
        Eigen::Map<Eigen::MatrixX<SReal>> EigenW(w[0],dimension, dimension) ;
        SReal eigenRadius = 0;
        for(auto s : EigenW.eigenvalues())
        {
            eigenRadius=std::max(eigenRadius,norm(s));
        }
        rho = d_spectralCorrectionFactor.getValue()*std::min(1.0, 0.9 * 2/eigenRadius);
    }

    for(int i=0; i<problem->maxIterations; i++)
    {
        iterCount ++;
        bool constraintsAreVerified = true;

        error=0.0;
        SReal beta = d_useConjugateResidue.getValue() * std::min(1.0, pow( ((float)i)/problem->maxIterations,d_conjugateResidueSpeedFactor.getValue()));

        for(int j=0; j<dimension; ) // increment of j realized at the end of the loop
        {
            // 1. nbLines provide the dimension of the constraint
            const unsigned int nb = problem->constraintsResolutions[j]->getNbLines();

            for(unsigned l=j; l<j+nb; ++l )
            {
                for(unsigned k=0; k<dimension; ++k)
                {
                    d[l] +=  w[l][k] * deltaF[k];
                }
                correctedD[l] = rho * d[l]  ;
            }
            j += nb;
        }

        for(int j=0; j<dimension; ) // increment of j realized at the end of the loop
        {
            // 1. nbLines provide the dimension of the constraint
            const unsigned int nb = problem->constraintsResolutions[j]->getNbLines();

            problem->constraintsResolutions[j]->resolution(j,w,correctedD.data(), force, dfree);
            for(unsigned l=j; l<j+nb; ++l )
            {
                force[l] += beta * deltaF[l] ;
                deltaF[l] = force[l] - lastF[l];
                lastF[l] = force[l];
            }

            SReal cstError = 0.0;
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

        if (problem->allVerified)
        {
            if (constraintsAreVerified)
            {
                convergence = true;
                break;
            }
        }
        else if(error < tol && i > 0) // do not stop at the first iteration (that is used for initial guess computation)
        {
            convergence = true;
            break;
        }
    }

    sofa::helper::AdvancedTimer::valSet("GS iterations", problem->currentIterations);

    problem->result_output(this, force, error, iterCount, convergence);

}



void registerImprovedJacobiConstraintSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("A Constraint Solver using the Linear Complementarity Problem formulation to solve Constraint based components using a Projected Jacobi iterative method")
        .add< ImprovedJacobiConstraintSolver >());
}


}