/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_CGLINEARSOLVER_INL
#define SOFA_COMPONENT_LINEARSOLVER_CGLINEARSOLVER_INL

#include <SofaBaseLinearSolver/CGLinearSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/helper/AdvancedTimer.h>

#include <sofa/core/ObjectFactory.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace linearsolver
{

/// Linear system solver using the conjugate gradient iterative algorithm
template<class TMatrix, class TVector>
CGLinearSolver<TMatrix,TVector>::CGLinearSolver()
    : f_maxIter( initData(&f_maxIter,(unsigned)25,"iterations","maximum number of iterations of the Conjugate Gradient solution") )
    , f_tolerance( initData(&f_tolerance,(SReal)1e-5,"tolerance","desired precision of the Conjugate Gradient Solution (ratio of current residual norm over initial residual norm)") )
    , f_smallDenominatorThreshold( initData(&f_smallDenominatorThreshold,(SReal)1e-5,"threshold","minimum value of the denominator in the conjugate Gradient solution") )
    , f_warmStart( initData(&f_warmStart,false,"warmStart","Use previous solution as initial solution") )
    , f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , f_graph( initData(&f_graph,"graph","Graph of residuals at each iteration") )
{
    f_graph.setWidget("graph");
//    f_graph.setReadOnly(true);
#ifdef DISPLAY_TIME
    timeStamp = 1.0 / (SReal)sofa::helper::system::thread::CTime::getRefTicksPerSec();
#endif

    f_maxIter.setRequired(true);
    f_tolerance.setRequired(true);
    f_smallDenominatorThreshold.setRequired(true);
}

template<class TMatrix, class TVector>
void CGLinearSolver<TMatrix,TVector>::resetSystem()
{
    f_graph.beginEdit()->clear();
    f_graph.endEdit();

    Inherit::resetSystem();
}

template<class TMatrix, class TVector>
void CGLinearSolver<TMatrix,TVector>::setSystemMBKMatrix(const sofa::core::MechanicalParams* mparams)
{
#ifdef DISPLAY_TIME
    sofa::helper::system::thread::CTime timer;
    time2 = (SReal) timer.getTime();
#endif

    Inherit::setSystemMBKMatrix(mparams);

#ifdef DISPLAY_TIME
    time2 = ((SReal) timer.getTime() - time2)  * timeStamp;
#endif
}

/// Solve Mx=b
template<class TMatrix, class TVector>
void CGLinearSolver<TMatrix,TVector>::solve(Matrix& M, Vector& x, Vector& b)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printComment("ConjugateGradient");
#endif

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("VectorAllocation");
#endif
    const core::ExecParams* params = core::ExecParams::defaultInstance();
    typename Inherit::TempVectorContainer vtmp(this, params, M, x, b);
    Vector& p = *vtmp.createTempVector();
    Vector& q = *vtmp.createTempVector();
    Vector& r = *vtmp.createTempVector();

    const bool verbose  = f_verbose.getValue();

    // -- solve the system using a conjugate gradient solution
    double rho, rho_1=0, alpha, beta;

    msg_info_when(verbose) <<" CGLinearSolver, b = "<< b ;


    if( f_warmStart.getValue() )
    {
        r = M * x;
        r.eq( b, r, -1.0 );   //  initial residual r = b - Ax;
    }
    else
    {
        x.clear();
        r = b; // initial residual
    }

    double normb = b.norm();
    std::map < std::string, sofa::helper::vector<SReal> >& graph = *f_graph.beginEdit();
    sofa::helper::vector<SReal>& graph_error = graph[(this->isMultiGroup()) ? this->currentNode->getName()+std::string("-Error") : std::string("Error")];
    graph_error.clear();
    sofa::helper::vector<SReal>& graph_den = graph[(this->isMultiGroup()) ? this->currentNode->getName()+std::string("-Denominator") : std::string("Denominator")];
    graph_den.clear();
    graph_error.push_back(1);
    unsigned nb_iter;
    const char* endcond = "iterations";

#ifdef DISPLAY_TIME
    sofa::helper::system::thread::CTime timer;
    time1 = (SReal) timer.getTime();
#endif

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("VectorAllocation");
#endif
    for( nb_iter=1; nb_iter<=f_maxIter.getValue(); nb_iter++ )
    {
#ifdef SOFA_DUMP_VISITOR_INFO
        std::ostringstream comment;
        if (simulation::Visitor::isPrintActivated())
        {
            comment << "Iteration_" << nb_iter;
            simulation::Visitor::printNode(comment.str());
        }
#endif
        // 		printWithElapsedTime( x, helper::system::thread::CTime::getTime()-time0,sout );

        //z = r; // no precond
        //rho = r.dot(z);
        rho = r.dot(r);

        if (nb_iter>1)
        {
            double normr = sqrt(rho); //sqrt(r.dot(r));
            double err = normr/normb;
            graph_error.push_back(err);
            if (err <= f_tolerance.getValue())
            {
                endcond = "tolerance";

#ifdef SOFA_DUMP_VISITOR_INFO
                if (simulation::Visitor::isPrintActivated())
                    simulation::Visitor::printCloseNode(comment.str());
#endif
                break;
            }
        }

        if( nb_iter==1 )
            p = r; //z;
        else
        {
            beta = rho / rho_1;
            //p = p*beta + r; //z;
            cgstep_beta(params, p,r,beta);
        }

        if( verbose )
        {
            sout<<"p : "<<p<<sendl;
        }

        // matrix-vector product
        q = M*p;

        if( verbose )
        {
            sout<<"q = M p : "<<q<<sendl;
        }

        double den = p.dot(q);

        graph_den.push_back(den);

        if( fabs(den)<f_smallDenominatorThreshold.getValue() )
        {
            endcond = "threshold";
            if( verbose )
            {
                sout<<"CGLinearSolver, den = "<<den<<", smallDenominatorThreshold = "<<f_smallDenominatorThreshold.getValue()<<sendl;
            }
#ifdef SOFA_DUMP_VISITOR_INFO
            if (simulation::Visitor::isPrintActivated())
                simulation::Visitor::printCloseNode(comment.str());
#endif
            break;
        }
        alpha = rho/den;
        //x.peq(p,alpha);                 // x = x + alpha p
        //r.peq(q,-alpha);                // r = r - alpha q
        cgstep_alpha(params, x,r,p,q,alpha);
        if( verbose )
        {
            sout<<"den = "<<den<<", alpha = "<<alpha<<sendl;
            sout<<"x : "<<x<<sendl;
            sout<<"r : "<<r<<sendl;
        }

        rho_1 = rho;
#ifdef SOFA_DUMP_VISITOR_INFO
        if (simulation::Visitor::isPrintActivated())
            simulation::Visitor::printCloseNode(comment.str());
#endif
    }

#ifdef DISPLAY_TIME
    time1 = (SReal)(((SReal) timer.getTime() - time1) * timeStamp / (nb_iter-1));
#endif

    f_graph.endEdit();

    sofa::helper::AdvancedTimer::valSet("CG iterations", nb_iter);

    // x is the solution of the system
#ifdef DISPLAY_TIME
    dmsg_info() << " solve, CG = "<<time1<<" build = "<< time2 ;
#endif

    dmsg_info() << "solve, nbiter = "<<nb_iter<<" stop because of "<<endcond ;
    dmsg_info_when( verbose ) <<"solve, solution = "<< x ;

    vtmp.deleteTempVector(&p);
    vtmp.deleteTempVector(&q);
    vtmp.deleteTempVector(&r);
}

template<class TMatrix, class TVector>
inline void CGLinearSolver<TMatrix,TVector>::cgstep_beta(const core::ExecParams* /*params*/, Vector& p, Vector& r, SReal beta)
{
    p *= beta;
    p += r; //z;
}

template<class TMatrix, class TVector>
inline void CGLinearSolver<TMatrix,TVector>::cgstep_alpha(const core::ExecParams* /*params*/, Vector& x, Vector& r, Vector& p, Vector& q, SReal alpha)
{
    x.peq(p,alpha);                 // x = x + alpha p
    r.peq(q,-alpha);                // r = r - alpha q
}

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_LINEARSOLVER_CGLINEARSOLVER_INL
