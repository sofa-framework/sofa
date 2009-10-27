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
// Author: Hadrien Courtecuisse
//
// Copyright: See COPYING file that comes with this distribution
#include <sofa/component/linearsolver/PCGLinearSolver.h>
#include <sofa/component/linearsolver/NewMatMatrix.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/componentmodel/behavior/LinearSolver.h>
#include <sofa/helper/system/thread/CTime.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;
using namespace sofa::simulation;
using namespace sofa::core::objectmodel;
using sofa::helper::system::thread::CTime;
using sofa::helper::system::thread::ctime_t;
using std::cerr;
using std::endl;

template<class TMatrix, class TVector>
void PCGLinearSolver<TMatrix,TVector>::init()
{
    std::vector<sofa::core::componentmodel::behavior::LinearSolver*> solvers;
    BaseContext * c = this->getContext();
    c->get<LinearSolver>(&solvers,BaseContext::SearchDown);

    for (unsigned int i=0; i<solvers.size(); ++i)
    {
        if (solvers[i] != this)
        {
            this->preconditioners.push_back(solvers[i]);
        }
    }

#ifdef DISPLAY_TIME
    time3 = 0.0;
    time1 = 0.0;
    time2 = 0.0;
    time4 = 0.0;
    step_simu=0;
    it_simu=0;
#endif

}

template<class TMatrix, class TVector>
void PCGLinearSolver<TMatrix,TVector>::setSystemMBKMatrix(double mFact, double bFact, double kFact)
{
    Inherit::setSystemMBKMatrix(mFact,bFact,kFact);

#ifdef DISPLAY_TIME
    double t3 = (double) CTime::getTime();
#endif

    no_precond = use_precond.getValue();

    if (no_precond)
    {
        if (iteration<=0)
        {
            for (unsigned int i=0; i<this->preconditioners.size(); ++i)
            {
                preconditioners[i]->setSystemMBKMatrix(mFact,bFact,kFact);
            }
            iteration = f_refresh.getValue();
        }
        else
        {
            iteration--;
        }
    }

#ifdef DISPLAY_TIME
    time3 += (double) CTime::getTime() - t3;
#endif

}

template<class TMatrix, class TVector>
void PCGLinearSolver<TMatrix,TVector>::solve (Matrix& M, Vector& x, Vector& b)
{
    using std::cerr;
    using std::endl;

    Vector& p = *this->createVector();
    Vector& q = *this->createVector();
    Vector& r = *this->createVector();
    Vector& z = *this->createVector();

    const bool printLog = f_printLog.getValue();
    const bool verbose  = f_verbose.getValue();

    // -- solve the system using a conjugate gradient solution
    double rho, rho_1=0, alpha, beta;

    if( verbose )
        cerr<<"PCGLinearSolver, b = "<< b <<endl;

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
    double tmp3 = (double) CTime::getTime();
    double tmp2 = 0.0;
    double tmp = 0.0;
#endif

    for (unsigned int i=0; i<this->preconditioners.size(); i++)
    {
        preconditioners[i]->setSystemLHVector(z);
        preconditioners[i]->setSystemRHVector(r);
        preconditioners[i]->invertSystem();
    }

#ifdef DISPLAY_TIME
    time4 += ((double) CTime::getTime() - tmp3);
    tmp3 = (double) CTime::getTime();
#endif

    for( nb_iter=1; nb_iter<=f_maxIter.getValue(); nb_iter++ )
    {

#ifdef SOFA_DUMP_VISITOR_INFO
        std::ostringstream comment;
        comment << "Iteration : " << nb_iter;
        simulation::Visitor::printComment(comment.str());
#endif

        if (this->preconditioners.size()==0 || (!no_precond))
        {
            z = r;
        }
        else
        {

            for (unsigned int i=0; i<this->preconditioners.size(); i++)
            {
#ifdef DISPLAY_TIME
                tmp2 = (double) CTime::getTime();
#endif
                preconditioners[i]->setSystemLHVector(z);
                preconditioners[i]->setSystemRHVector(r);
                preconditioners[i]->solveSystem();
#ifdef DISPLAY_TIME
                tmp += ((double) CTime::getTime() - tmp2);
#endif
            }
        }

        rho = r.dot(z);

        if (nb_iter>1)
        {
            double normr = sqrt(r.dot(r));
            double err = normr/normb;
            graph_error.push_back(err);
            if (err <= f_tolerance.getValue())
            {
                endcond = "tolerance";
                break;
            }
        }

        if( nb_iter==1 ) p = z;
        else
        {
            beta = rho / rho_1;
            //p = p*beta + z;
            cgstep_beta(p,z,beta);
        }

        if( verbose )
        {
            cerr<<"p : "<<p<<endl;
        }

        // matrix-vector product
        q = M*p;

        if( verbose )
        {
            cerr<<"q = M p : "<<q<<endl;
        }

        double den = p.dot(q);

        graph_den.push_back(den);

        if( fabs(den)<f_smallDenominatorThreshold.getValue() )
        {
            endcond = "threshold";
            if( verbose )
            {
                cerr<<"PCGLinearSolver, den = "<<den<<", smallDenominatorThreshold = "<<f_smallDenominatorThreshold.getValue()<<endl;
            }
            break;
        }

        alpha = rho/den;
        //x.peq(p,alpha);                 // x = x + alpha p
        //r.peq(q,-alpha);                // r = r - alpha q
        cgstep_alpha(x,r,p,q,alpha);

        if( verbose )
        {
            cerr<<"den = "<<den<<", alpha = "<<alpha<<endl;
            cerr<<"x : "<<x<<endl;
            cerr<<"r : "<<r<<endl;
        }

        rho_1 = rho;

        //printf("%f\n",(CTime::getRefTime() - time1)  / (double)CTime::getRefTicksPerSec());

    }

#ifdef DISPLAY_TIME
    time1 += ((double) CTime::getTime() - tmp3) - tmp;
    time2 += tmp;
#endif

    f_graph.endEdit();
    // x is the solution of the system
#ifdef DISPLAY_TIME
    step_simu++;
    it_simu+=nb_iter;
    if (step_simu>DISPLAY_TIME)
    {
        time1 /= (double)((double)CTime::getRefTicksPerSec());
        time2 /= (double)((double)CTime::getRefTicksPerSec());
        time3 /= (double)((double)CTime::getRefTicksPerSec());
        time4 /= (double)((double)CTime::getRefTicksPerSec());

        double total = time1+time2+time4+time3;
        double percen = 100.0/total;
        cerr<<"\nPCGLinearSolver::solve nbiter = "<<it_simu<<" total time = "<<total<<"\nCG =\t\t("<<time1<<"\t"<<(time1*percen)<<"%)\npreconditioner =("<<time2<<"\t"<<(time2*percen)<<"%)\nInvert =\t("<<time4<<"\t"<<(time4*percen)<<"%)\nbuild =\t\t("<<time3<<"\t"<<(time3*percen)<<"%)"<<endl;

        time1 = 0.0;
        time2 = 0.0;
        time3 = 0.0;
        time4 = 0.0;
        step_simu=0;
        it_simu=0;
    }
#endif
    if( printLog )
    {
        cerr<<"PCGLinearSolver::solve, nbiter = "<<nb_iter<<" stop because of "<<endcond<<endl;
    }
    if( verbose )
    {
        cerr<<"PCGLinearSolver::solve, solution = "<<x<<endl;
    }
    this->deleteVector(&p);
    this->deleteVector(&q);
    this->deleteVector(&r);
    this->deleteVector(&z);

}

SOFA_DECL_CLASS(PCGLinearSolver)

int PCGLinearSolverClass = core::RegisterObject("Linear system solver using the conjugate gradient iterative algorithm")
        .add< PCGLinearSolver<GraphScatteredMatrix,GraphScatteredVector> >(true)
//.add< PCGLinearSolver<NewMatMatrix,NewMatVector> >()
//.add< PCGLinearSolver<NewMatSymmetricMatrix,NewMatVector> >()
//.add< PCGLinearSolver<NewMatBandMatrix,NewMatVector> >()
//.add< PCGLinearSolver<NewMatSymmetricBandMatrix,NewMatVector> >()
//.add< PCGLinearSolver< FullMatrix<double>, FullVector<double> > >()
//.add< PCGLinearSolver< SparseMatrix<double>, FullVector<double> > >()
        .addAlias("PCGSolver")
        .addAlias("PConjugateGradient")
        ;

} // namespace linearsolver

} // namespace component

} // namespace sofa

