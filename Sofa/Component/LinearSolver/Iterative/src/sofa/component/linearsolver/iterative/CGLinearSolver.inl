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
#pragma once
#include <sofa/component/linearsolver/iterative/CGLinearSolver.h>
#include <sofa/simulation/MechanicalVisitor.h>

#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
using sofa::helper::ScopedAdvancedTimer ;

namespace sofa::component::linearsolver::iterative
{

/// Linear system solver using the conjugate gradient iterative algorithm
template<class TMatrix, class TVector>
CGLinearSolver<TMatrix,TVector>::CGLinearSolver()
    : d_maxIter( initData(&d_maxIter, 25u,"iterations","Maximum number of iterations of the Conjugate Gradient solution") )
    , d_tolerance( initData(&d_tolerance,(Real)1e-5,"tolerance","Desired accuracy of the Conjugate Gradient solution evaluating: |r|²/|b|² (ratio of current residual norm over initial residual norm)") )
    , d_smallDenominatorThreshold( initData(&d_smallDenominatorThreshold,(Real)1e-5,"threshold","Minimum value of the denominator (pT A p)^ in the conjugate Gradient solution") )
    , d_warmStart( initData(&d_warmStart,false,"warmStart","Use previous solution as initial solution") )
    , d_graph( initData(&d_graph,"graph","Graph of residuals at each iteration") )
{
    d_graph.setWidget("graph");
    d_maxIter.setRequired(true);
    d_tolerance.setRequired(true);
    d_smallDenominatorThreshold.setRequired(true);
}

/// Initialization function checking input Data
template<class TMatrix, class TVector>
void CGLinearSolver<TMatrix,TVector>::init()
{
    Inherit1::init();

    if(d_tolerance.getValue() < 0.0)
    {
        msg_warning() << "'tolerance' must be a positive value" << msgendl
                      << "default value used: 1e-5";
        d_tolerance.setValue(1e-5);
    }
    if(d_smallDenominatorThreshold.getValue() < 0.0)
    {
        msg_warning() << "'threshold' must be a positive value" << msgendl
                      << "default value used: 1e-5";
        d_smallDenominatorThreshold.setValue(1e-5);
    }

    timeStepCount = 0;
    equilibriumReached = false;
}

/// Clear graph and clean the RHS / LHS vectors
template<class TMatrix, class TVector>
void CGLinearSolver<TMatrix,TVector>::resetSystem()
{
    d_graph.beginEdit()->clear();
    d_graph.endEdit();

    Inherit::resetSystem();
}

/// For unbuilt approach (e.g. with GraphScattered types),
/// it passes the coefficients multiplying the matrices M, B and K from the ODE to the LinearSolver (MechanicalOperations::setKFactor) and includes a resetSystem
/// In other cases
/// the global system matrix is setup (pass coefficients with MechanicalOperations::setKFactor) and built it by calling the addMBKToMatrix visitor
template<class TMatrix, class TVector>
void CGLinearSolver<TMatrix,TVector>::setSystemMBKMatrix(const sofa::core::MechanicalParams* mparams)
{
    SCOPED_TIMER("CG-setSystemMBKMatrix");
    Inherit::setSystemMBKMatrix(mparams);
}

/// Solve iteratively the linear system Ax=b following a conjugate gradient descent
template<class TMatrix, class TVector>
void CGLinearSolver<TMatrix,TVector>::solve(Matrix& A, Vector& x, Vector& b)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printComment("ConjugateGradient");
#endif

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("VectorAllocation");
#endif

    /// Allocate the required vectors for the iterative resolution
    const core::ExecParams* params = core::execparams::defaultInstance();
    typename Inherit::TempVectorContainer vtmp(this, params, A, x, b);
    Vector& p = *vtmp.createTempVector(); // orthogonal directions
    Vector& q = *vtmp.createTempVector(); // temporary vector computing A*p
    Vector& r = *vtmp.createTempVector(); // residual

    Real rho, rho_1=0, alpha, beta;

    msg_info() << "b = " << b ;

    /// Compute the initial residual r depending on the warmStart option
    if( d_warmStart.getValue() )
    {
        r = A * x;
        r.eq( b, r, -1.0 );   // initial residual r = b - Ax;
    }
    else
    {
        x.clear();
        r = b;                // initial residual r = b
    }

    /// Compute the norm of the right-hand-side vector b
    const auto normb = b.norm();

    std::map < std::string, sofa::type::vector<Real> >& graph = *d_graph.beginEdit();
    sofa::type::vector<Real>& graph_error = graph[std::string("Error")];
    graph_error.clear();
    graph_error.push_back(1);

    sofa::type::vector<Real>& graph_den = graph[std::string("Denominator")];
    graph_den.clear();


    unsigned nb_iter = 0;
    const char* endcond = "iterations";

    sofa::helper::AdvancedTimer::stepBegin("CG-Solve");

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("VectorAllocation");
#endif

    // Check if forces in the Left Hand Side (LHS) vector are non-zero
    if(normb != 0.0)
    {
        for( nb_iter = 1; nb_iter <= d_maxIter.getValue(); nb_iter++ )
        {
#ifdef SOFA_DUMP_VISITOR_INFO
            std::ostringstream comment;
            if (simulation::Visitor::isPrintActivated())
            {
                comment << "Iteration_" << nb_iter;
                simulation::Visitor::printNode(comment.str());
            }
#endif

            /// Compute ρ = r²
            rho = r.dot(r);

            /// Compute the error from the norm of ρ and b
            const auto normr = sqrt(rho);
            const auto err = normr/normb;
            assert(!std::isnan(err));

            graph_error.push_back(err);


            /// Break condition = TOLERANCE criterion regarding the error err=|r|²/|b|² is reached
            if (err <= d_tolerance.getValue())
            {
                /// Tolerance met at first step, tolerance value might not be relevant
                if(nb_iter == 1 && timeStepCount == 0)
                {
                    msg_warning() << "tolerance reached at first iteration of CG" << msgendl
                                  << "Check the 'tolerance' data field, you might decrease it";
                }
                else
                {
                    if(nb_iter == 1 && !equilibriumReached)
                    {
                        msg_info() << "Equilibrium reached regarding tolerance";
                        equilibriumReached = true;
                    }
                    if(nb_iter > 1)
                    {
                        equilibriumReached = false;
                    }

                    endcond = "tolerance";
                    msg_info() << "error = " << err <<", tolerance = " << d_tolerance.getValue();

#ifdef SOFA_DUMP_VISITOR_INFO
                    if (simulation::Visitor::isPrintActivated())
                        simulation::Visitor::printCloseNode(comment.str());
#endif
                    break;
                }
            }


            /// Compute the value of p, conjugate with x
            if( nb_iter==1 )    // FIRST step:      p = r
            {
                p = r;
            }
            else                // ALL other steps: p = r + beta * p
            {
                beta = rho / rho_1;

                /// Compute the next conjugate direction p for iteration "nb_iter"
                /// p = r + p*beta
                cgstep_beta(params, p,r,beta);
            }

            msg_info() << "p : " << p;

            /// Compute the matrix-vector product A p to compute the denominator
            /// This matrix-vector product depends on the type of matrix:
            /// 1) The matrix is assembled (e.g. CompressedRowSparseMatrix): traditional matrix-vector product
            /// 2) The matrix is not assembled (e.g. GraphScattered): visitors run and call addMBKdx on force
            /// fields (usually force fields implement addDForce). This method performs the matrix-vector product and
            /// store it in another vector without building explicitly the matrix. Projective constraints are also applied.
            q = A*p;
            msg_info() << "q = A p : " << q;

            /// Compute the denominator : pT A p
            const auto den = p.dot(q);

            graph_den.push_back(den);

            if(den != 0.0) // as a denominator, we need to check if not zero else division will return the infinite value
            {
                /// Break condition = THRESHOLD criterion regarding the denominator pT A p is reached (but do at least one iteration)
                if (fabs(den) <= d_smallDenominatorThreshold.getValue())
                {
                    /// Threshold met at first step, threshold value might not be relevant
                    if(nb_iter == 1 && timeStepCount == 0)
                    {
                        msg_warning() << "denominator threshold reached at first iteration of CG" << msgendl
                                      << "Check the 'threshold' data field, you might decrease it";
                    }
                    else
                    {
                        if(nb_iter == 1 && !equilibriumReached)
                        {
                            msg_info() << "Equilibrium reached regarding threshold";
                            equilibriumReached = true;
                        }
                        if(nb_iter > 1)
                        {
                            equilibriumReached = false;
                        }

                        endcond = "threshold";
                        msg_info() << "den = " << den <<", smallDenominatorThreshold = " << d_smallDenominatorThreshold.getValue() <<", err = " << err;

#ifdef SOFA_DUMP_VISITOR_INFO
                    if (simulation::Visitor::isPrintActivated())
                        simulation::Visitor::printCloseNode(comment.str());
#endif
                        break;
                    }
                }


                /// Compute the coefficient α for the conjugate direction
                alpha = rho/den;

                /// End of the CG step by updating x and r
                /// x = x + alpha p
                /// r = r - alpha p
                cgstep_alpha(params, x,r,p,q,alpha);

                msg_info() << "den = " << den << ", alpha = " << alpha << ", x = " << x << ", r = " << r;
            }
            else
            {
                msg_warning() << "den = 0.0, break the iterations";
                break;
            }

            rho_1 = rho;

#ifdef SOFA_DUMP_VISITOR_INFO
            if (simulation::Visitor::isPrintActivated())
                simulation::Visitor::printCloseNode(comment.str());
#endif
        }
    }
    // Case no forces applied, b=0
    else
    {
        endcond = "null norm of vector b";

        // If first step : check the value of threshold
        if( timeStepCount==0 )
        {
            p = r;
            q = A*p;
            const auto den = p.dot(q);

            if(den != 0.0)
            {
                if (fabs(den) <= d_smallDenominatorThreshold.getValue())
                {
                    msg_warning() << "denominator threshold reached at first iteration of CG" << msgendl
                                  << "Check the 'threshold' data field, you might decrease it";
                }
            }
            else
            {
                msg_info() << "no way to check the validity of : tolerance and threshold value";
            }
        }
    }

    sofa::helper::AdvancedTimer::stepEnd("CG-Solve");

    d_graph.endEdit();
    timeStepCount ++;

    sofa::helper::AdvancedTimer::valSet("CG iterations", nb_iter);

    msg_info() << "solve, nbiter = "<<nb_iter<<" stop because of "<<endcond;
    msg_info() <<"solve, solution = "<< x ;

    /// Delete all temporary vectors p, q and r
    vtmp.deleteTempVector(&p);
    vtmp.deleteTempVector(&q);
    vtmp.deleteTempVector(&r);
}

template<class TMatrix, class TVector>
inline void CGLinearSolver<TMatrix,TVector>::cgstep_beta(const core::ExecParams* /*params*/, Vector& p, Vector& r, Real beta)
{
    // p = p*beta + r
    p *= beta;
    p += r;
}

template<class TMatrix, class TVector>
inline void CGLinearSolver<TMatrix,TVector>::cgstep_alpha(const core::ExecParams* /*params*/, Vector& x, Vector& r, Vector& p, Vector& q, Real alpha)
{
    // x = x + alpha p
    x.peq(p,alpha);

    // r = r - alpha q
    r.peq(q,-alpha);
}

} // namespace sofa::component::linearsolver::iterative
