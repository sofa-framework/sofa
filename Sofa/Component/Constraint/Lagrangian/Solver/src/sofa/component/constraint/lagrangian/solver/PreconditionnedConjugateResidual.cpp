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

#include <sofa/component/constraint/lagrangian/solver/PreconditionnedConjugateResidual.h>
#include <sofa/component/constraint/lagrangian/solver/GenericConstraintSolver.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/core/ObjectFactory.h>
#include <Eigen/Eigenvalues>

namespace sofa::component::constraint::lagrangian::solver
{

PreconditionnedConjugateResidual::PreconditionnedConjugateResidual()
    : BuiltConstraintSolver()
{}

typedef std::vector<std::vector<SReal>> BufferMatrix;
typedef std::vector<SReal> BufferVector;

inline void bufferMatrixVectorMult(BufferMatrix& W, const BufferVector& vec, BufferVector& result)
{
    const unsigned dimension = W.size();
    for(unsigned j=0; j< dimension ; ++j)
    {
        result[j] = 0;
        for(unsigned k=0; k< dimension; ++k)
        {
            result[j]+= W[j][k] * vec[k];
        }
    }
}

inline void bufferVectorAdd(const BufferVector& vec1, const BufferVector& vec2, const SReal fact2,  BufferVector& result)
{
    const unsigned dimension = vec1.size();
        for(unsigned j=0; j< dimension; ++j)
    {
        result[j] = vec1[j] + fact2 * vec2[j];
    }
}
inline void bufferVectorAdd(const SReal* vec1, const BufferVector& vec2, const SReal fact2,  SReal *  result)
{
    const unsigned dimension = vec2.size();
        for(unsigned j=0; j< dimension; ++j)
    {
        result[j] = vec1[j] + fact2 * vec2[j];
    }
}

inline void bufferVectorSubtract(const BufferVector& vec1, const BufferVector& vec2, const SReal fact2,  BufferVector& result)
{
    const unsigned dimension = vec1.size();
    for(unsigned j=0; j< dimension; ++j)
    {
        result[j] = vec1[j] - fact2 * vec2[j];
    }
}

inline SReal bufferVectorDotProduct(const BufferVector& vec1, BufferVector& vec2)
{
    const unsigned dimension = vec1.size();
    SReal result = 0;
    for (unsigned j=0; j< dimension; ++j)
    {
        result += vec1[j] * vec2[j];
    }
    return result;
}


void PreconditionnedConjugateResidual::doSolve(GenericConstraintProblem * problem , SReal timeout)
{
    SCOPED_TIMER_VARNAME(gaussSeidelTimer, "PreconditionnedConjugateResidual");


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
    std::vector<SReal> r(dimension);
    std::vector<SReal> p(dimension);
    std::vector<SReal> Wp(dimension);
    std::vector<SReal> Wr(dimension);
    std::vector<std::vector<SReal>> MW(dimension, std::vector<SReal>(dimension));

    SReal Beta = 0.0;


    //Apply Jacobi preconditioner
    for(unsigned j=0; j< dimension; ++j)
    {
        const SReal invWjj = 1.0/w[j][j];
        for(unsigned k=0; k< dimension; ++k)
        {
            MW[j][k] = w[j][k] * invWjj ;
        }
    }
    memcpy(r.data(), dfree, dimension*sizeof(SReal));
    for(unsigned j=0; j< dimension; ++j)
    {
        const SReal invWjj = 1.0/w[j][j];
        r[j] *= invWjj;
    }

    memset(force, 0, dimension*sizeof(SReal));
    std::copy_n(r.cbegin(), dimension, p.begin());
    bufferMatrixVectorMult(MW, p, Wp);
    bufferMatrixVectorMult(MW, r, Wr);
    SReal rWr = bufferVectorDotProduct(r, Wr);


    SReal error=0.0;
    bool convergence = false;
    if(problem->scaleTolerance && !problem->allVerified)
    {
        tol *= dimension;
    }

    const unsigned maxIt = std::min(problem->maxIterations, problem->getDimension());
    unsigned iterCount = 0;
    for(unsigned i=0; i<maxIt; i++)
    {
        iterCount ++;
        bool constraintsAreVerified = true;

        error=0.0;

        // Alpha computation
        const SReal alpha = rWr / bufferVectorDotProduct(Wp, Wp);

        // Unknown update
        bufferVectorAdd(force, p, alpha, force);

        // Residue update
        bufferVectorSubtract(r, Wp, alpha, r);

        // W*residue update
        bufferMatrixVectorMult(MW, r, Wr);
        const double oldrWr = rWr;
        rWr = bufferVectorDotProduct(r, Wr);

        // Beta computation
        const SReal beta = rWr / oldrWr;

        // p update
        bufferVectorAdd(r, p, beta, p);

        for(int j=0; j<dimension; ) // increment of j realized at the end of the loop
        {
            // 1. nbLines provide the dimension of the constraint
            const unsigned int nb = problem->constraintsResolutions[j]->getNbLines();
            SReal cstError = 0.0;
            for(unsigned l=j; l<j+nb; ++l )
            {
                cstError += pow(r[i],2);
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


        // W*p update
        bufferVectorAdd(Wr, Wp, beta, Wp);

    }

    sofa::helper::AdvancedTimer::valSet("PCR iterations", problem->currentIterations);

    problem->result_output(this, force, error, iterCount, convergence);

}



void registerPreconditionnedConjugateResidual(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("A Constraint Solver using the Linear Complementarity Problem formulation to solve Constraint based components using a Projected Jacobi iterative method")
        .add< PreconditionnedConjugateResidual >());
}


}