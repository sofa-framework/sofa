/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_LINEARSOLVER_CGLINEARSOLVER_H
#define SOFA_COMPONENT_LINEARSOLVER_CGLINEARSOLVER_H

#include <sofa/core/componentmodel/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/MatrixLinearSolver.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/helper/map.h>
#include <math.h>


namespace sofa
{

namespace component
{

namespace linearsolver
{

//#define DISPLAY_TIME

#ifdef DISPLAY_TIME
#include <sofa/helper/system/thread/CTime.h>
using sofa::helper::system::thread::CTime;
#endif


/// Linear system solver using the conjugate gradient iterative algorithm
template<class TMatrix, class TVector>
class SOFA_COMPONENT_LINEARSOLVER_API CGLinearSolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector>, public virtual sofa::core::objectmodel::BaseObject
{
public:
    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;
    Data<unsigned> f_maxIter;
    Data<double> f_tolerance;
    Data<double> f_smallDenominatorThreshold;
    Data<bool> f_verbose;
    Data<std::map < std::string, sofa::helper::vector<double> > > f_graph;
#ifdef DISPLAY_TIME
    double time1;
    double time2;
    double timeStamp;
#endif

    CGLinearSolver()
        : f_maxIter( initData(&f_maxIter,(unsigned)25,"iterations","maximum number of iterations of the Conjugate Gradient solution") )
        , f_tolerance( initData(&f_tolerance,1e-5,"tolerance","desired precision of the Conjugate Gradient Solution (ratio of current residual norm over initial residual norm)") )
        , f_smallDenominatorThreshold( initData(&f_smallDenominatorThreshold,1e-5,"threshold","minimum value of the denominator in the conjugate Gradient solution") )
        , f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
        , f_graph( initData(&f_graph,"graph","Graph of residuals at each iteration") )
    {
        f_graph.setWidget("graph");
        f_graph.setReadOnly(true);
#ifdef DISPLAY_TIME
        timeStamp = 1.0 / (double)CTime::getRefTicksPerSec();
#endif

    }
protected:
    /// This method is separated from the rest to be able to use custom/optimized versions depending on the types of vectors.
    /// It computes: p = p*beta + r
    inline void cgstep_beta(Vector& p, Vector& r, double beta);
    /// This method is separated from the rest to be able to use custom/optimized versions depending on the types of vectors.
    /// It computes: x += p*alpha, r -= q*alpha
    inline void cgstep_alpha(Vector& x, Vector& r, Vector& p, Vector& q, double alpha);

public:
    /// Solve Mx=b
    void solve (Matrix& M, Vector& x, Vector& b)
    {


#ifdef SOFA_DUMP_VISITOR_INFO
        simulation::Visitor::printComment("ConjugateGradient");
#endif


#ifdef SOFA_DUMP_VISITOR_INFO
        simulation::Visitor::printNode("VectorAllocation");
#endif
        Vector& p = *this->createVector();
        Vector& q = *this->createVector();
        Vector& r = *this->createVector();

        const bool printLog = f_printLog.getValue();
        const bool verbose  = f_verbose.getValue();

        // -- solve the system using a conjugate gradient solution
        double rho, rho_1=0, alpha, beta;

        if( verbose )
            serr<<"CGLinearSolver, b = "<< b <<sendl;

        x.clear();
        r = b; // initial residual

        double normb2 = b.dot(b);
        double normb = sqrt(normb2);
        std::map < std::string, sofa::helper::vector<double> >& graph = *f_graph.beginEdit();
        sofa::helper::vector<double>& graph_error = graph["Error"];
        graph_error.clear();
        sofa::helper::vector<double>& graph_den = graph["Denominator"];
        graph_den.clear();
        graph_error.push_back(1);
        unsigned nb_iter;
        const char* endcond = "iterations";

#ifdef DISPLAY_TIME
        CTime * timer;
        time1 = (double) timer->getTime();
#endif

#ifdef SOFA_DUMP_VISITOR_INFO
        simulation::Visitor::printCloseNode("VectorAllocation");
#endif
        for( nb_iter=1; nb_iter<=f_maxIter.getValue(); nb_iter++ )
        {
#ifdef SOFA_DUMP_VISITOR_INFO
            std::ostringstream comment;
            comment << "Iteration_" << nb_iter;
            simulation::Visitor::printNode(comment.str());
#endif
            // 		printWithElapsedTime( x, helper::system::thread::CTime::getTime()-time0,sout );

            //z = r; // no precond
            //rho = r.dot(z);
            rho = (nb_iter==1) ? normb2 : r.dot(r);

            if (nb_iter>1)
            {
                double normr = sqrt(rho); //sqrt(r.dot(r));
                double err = normr/normb;
                graph_error.push_back(err);
                if (err <= f_tolerance.getValue())
                {
                    endcond = "tolerance";

#ifdef SOFA_DUMP_VISITOR_INFO
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
                cgstep_beta(p,r,beta);
            }

            if( verbose )
            {
                serr<<"p : "<<p<<sendl;
            }

            // matrix-vector product
            q = M*p;

            if( verbose )
            {
                serr<<"q = M p : "<<q<<sendl;
            }

            double den = p.dot(q);

            graph_den.push_back(den);

            if( fabs(den)<f_smallDenominatorThreshold.getValue() )
            {
                endcond = "threshold";
                if( verbose )
                {
                    serr<<"CGLinearSolver, den = "<<den<<", smallDenominatorThreshold = "<<f_smallDenominatorThreshold.getValue()<<sendl;
                }
#ifdef SOFA_DUMP_VISITOR_INFO
                simulation::Visitor::printCloseNode(comment.str());
#endif
                break;
            }
            alpha = rho/den;
            //x.peq(p,alpha);                 // x = x + alpha p
            //r.peq(q,-alpha);                // r = r - alpha q
            cgstep_alpha(x,r,p,q,alpha);
            if( verbose )
            {
                serr<<"den = "<<den<<", alpha = "<<alpha<<sendl;
                serr<<"x : "<<x<<sendl;
                serr<<"r : "<<r<<sendl;
            }

            rho_1 = rho;
#ifdef SOFA_DUMP_VISITOR_INFO
            simulation::Visitor::printCloseNode(comment.str());
#endif
        }

#ifdef DISPLAY_TIME
        time1 = (double)(((double) timer->getTime() - time1) * timeStamp / (nb_iter-1));
#endif

        f_graph.endEdit();
        // x is the solution of the system
        if( printLog )
        {
#ifdef DISPLAY_TIME
            cerr<<"CGLinearSolver::solve, CG = "<<time1<<" bluid = "<<time2<<endl;
#endif
            serr<<"CGLinearSolver::solve, nbiter = "<<nb_iter<<" stop because of "<<endcond<<sendl;
        }
        if( verbose )
        {
            serr<<"CGLinearSolver::solve, solution = "<<x<<sendl;
        }
        this->deleteVector(&p);
        this->deleteVector(&q);
        this->deleteVector(&r);


    }

#ifdef DISPLAY_TIME

    void setSystemMBKMatrix(double mFact, double bFact, double kFact)
    {
        CTime * timer;
        time2 = (double) timer->getTime();

        Inherit::setSystemMBKMatrix(mFact,bFact,kFact);

        time2 = ((double) timer->getTime() - time2)  * timeStamp;
    }

#endif
};

template<class TMatrix, class TVector>
inline void CGLinearSolver<TMatrix,TVector>::cgstep_beta(Vector& p, Vector& r, double beta)
{
    p *= beta;
    p += r; //z;
}

template<class TMatrix, class TVector>
inline void CGLinearSolver<TMatrix,TVector>::cgstep_alpha(Vector& x, Vector& r, Vector& p, Vector& q, double alpha)
{
    x.peq(p,alpha);                 // x = x + alpha p
    r.peq(q,-alpha);                // r = r - alpha q
}

template<>
inline void CGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_beta(Vector& p, Vector& r, double beta)
{
    this->v_op(p,r,p,beta); // p = p*beta + r
}

template<>
inline void CGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_alpha(Vector& x, Vector& r, Vector& p, Vector& q, double alpha)
{
#ifdef SOFA_NO_VMULTIOP // unoptimized version
    x.peq(p,alpha);                 // x = x + alpha p
    r.peq(q,-alpha);                // r = r - alpha q
#else // single-operation optimization
    typedef core::componentmodel::behavior::BaseMechanicalState::VMultiOp VMultiOp;
    VMultiOp ops;
    ops.resize(2);
    ops[0].first = (VecId)x;
    ops[0].second.push_back(std::make_pair((VecId)x,1.0));
    ops[0].second.push_back(std::make_pair((VecId)p,alpha));
    ops[1].first = (VecId)r;
    ops[1].second.push_back(std::make_pair((VecId)r,1.0));
    ops[1].second.push_back(std::make_pair((VecId)q,-alpha));
    simulation::MechanicalVMultiOpVisitor vmop(ops);
    vmop.execute(this->getContext());
#endif
}

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
