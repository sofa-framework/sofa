/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_PARALLELCGLINEARSOLVER_H
#define SOFA_COMPONENT_LINEARSOLVER_PARALLELCGLINEARSOLVER_H
#include <SofaMisc/config.h>

#ifdef SOFA_SMP
#include <sofa/core/behavior/ParallelMultiVec.h>
using namespace sofa::defaulttype::SharedTypes;
#endif

#include <SofaBaseLinearSolver/MatrixLinearSolver.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

/// Linear system solver using the conjugate gradient iterative algorithm
template<class TMatrix, class TVector>
class SOFA_MISC_API ParallelCGLinearSolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector>
{

public:
    SOFA_CLASS(SOFA_TEMPLATE2(ParallelCGLinearSolver,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector));

//		typedef sofa::core::behavior::ParallelMultiVector<ParallelCGLinearSolver> MultiVector;
    // typedef ParallelOdeSolverImpl::MultiVector MultiVector;
    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef TVector MultiVector;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;

    ParallelCGLinearSolver();
    ~ParallelCGLinearSolver();

//    void solve (double dt);
    Data<unsigned> f_maxIter; ///< maximum number of iterations of the Conjugate Gradient solution
    Data<double> f_tolerance; ///< desired precision of the Conjugate Gradient Solution (ratio of current residual norm over initial residual norm)
    Data<double> f_smallDenominatorThreshold; ///< minimum value of the denominator in the conjugate Gradient solution
    Data<bool> f_verbose; ///< Dump system state at each iteration

    void resetSystem();

    // virtual void setSysteMBKMatrix(double mFact=0.0, double bFact=0.0, double kFact=0.0);

    // virtual void setSystemRHVector(VecID v);

    // virtual void setSystemLHVector(VecId v);

    // MatrixLinearSolver interface
    void solve(Matrix& M, Vector& x, Vector& b);
    // virtual void solveSystem();


protected:
    Shared<double> *rhoSh,*rho_1Sh,*alphaSh,*betaSh,*denSh,normbSh;
    Shared<bool> *breakCondition;
//    void cgLoop( MultiVector &x, MultiVector &r,MultiVector& p,MultiVector &q,const bool verbose);
};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
