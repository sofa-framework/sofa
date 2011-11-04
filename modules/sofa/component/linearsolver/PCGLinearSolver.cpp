/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/PCGLinearSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/linearsolver/NewMatMatrix.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
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

using namespace sofa::defaulttype;
using namespace sofa::core::behavior;
using namespace sofa::simulation;
using namespace sofa::core::objectmodel;
using sofa::helper::system::thread::CTime;
using sofa::helper::system::thread::ctime_t;
using std::cerr;
using std::endl;

template<class TMatrix, class TVector>
PCGLinearSolver<TMatrix,TVector>::PCGLinearSolver()
    : f_maxIter( initData(&f_maxIter,(unsigned)25,"iterations","maximum number of iterations of the Conjugate Gradient solution") )
    , f_tolerance( initData(&f_tolerance,1e-5,"tolerance","desired precision of the Conjugate Gradient Solution (ratio of current residual norm over initial residual norm)") )
    , f_smallDenominatorThreshold( initData(&f_smallDenominatorThreshold,1e-5,"threshold","minimum value of the denominator in the conjugate Gradient solution") )
    , f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , f_refresh( initData(&f_refresh,0,"refresh","Refresh iterations") )
    , use_precond( initData(&use_precond,true,"use_precond","Use preconditioners") )
    , f_preconditioners( initData(&f_preconditioners, "preconditioners", "If not empty: path to the solvers to use as preconditioners") )
    , f_graph( initData(&f_graph,"graph","Graph of residuals at each iteration") )
{
    f_graph.setWidget("graph");
//    f_graph.setReadOnly(true);
    iteration = 0;
    usePrecond = true;
}

template<class TMatrix, class TVector>
void PCGLinearSolver<TMatrix,TVector>::init()
{
    std::vector<sofa::core::behavior::LinearSolver*> solvers;
    BaseContext * c = this->getContext();

    const helper::vector<std::string>& precondNames = f_preconditioners.getValue();
    if (precondNames.empty() || !use_precond.getValue())
    {
        c->get<sofa::core::behavior::LinearSolver>(&solvers,BaseContext::SearchDown);
    }
    else
    {
        for (unsigned int i=0; i<precondNames.size(); ++i)
        {
            sofa::core::behavior::LinearSolver* s = NULL;
            c->get(s, precondNames[i]);
            if (s) solvers.push_back(s);
            else serr << "Solver \"" << precondNames[i] << "\" not found." << sendl;
        }
    }

    for (unsigned int i=0; i<solvers.size(); ++i)
    {
        if (solvers[i] && solvers[i] != this)
        {
            this->preconditioners.push_back(solvers[i]);
        }
    }

    sout<<"Found " << this->preconditioners.size() << " preconditioners"<<sendl;

    first = true;
}

template<class TMatrix, class TVector>
void PCGLinearSolver<TMatrix,TVector>::setSystemMBKMatrix(const core::MechanicalParams* mparams)
{
    sofa::helper::AdvancedTimer::valSet("PCG::buildMBK", 1);
    sofa::helper::AdvancedTimer::stepBegin("PCG::setSystemMBKMatrix");

    Inherit::setSystemMBKMatrix(mparams);

    sofa::helper::AdvancedTimer::stepEnd("PCG::setSystemMBKMatrix(Precond)");

    usePrecond = use_precond.getValue();

    if (first)   //We initialize all the preconditioners for the first step
    {
        first = false;
        if (iteration<=0)
        {
            for (unsigned int i=0; i<this->preconditioners.size(); ++i)
            {
                preconditioners[i]->setSystemMBKMatrix(mparams);
            }
            iteration = f_refresh.getValue();
        }
        else
        {
            iteration--;
        }
    }
    else if (usePrecond)     // We use only the first precond in the list
    {
        sofa::helper::AdvancedTimer::valSet("PCG::PrecondBuildMBK", 1);
        sofa::helper::AdvancedTimer::stepBegin("PCG::PrecondSetSystemMBKMatrix");

        if (iteration<=0)
        {
            preconditioners[0]->setSystemMBKMatrix(mparams);
            iteration = f_refresh.getValue();
        }
        else
        {
            iteration--;
        }
        sofa::helper::AdvancedTimer::stepEnd("PCG::PrecondSetSystemMBKMatrix");
    }


}

template<>
inline void PCGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_beta(const core::ExecParams* /*params*/ /* PARAMS FIRST */, Vector& p, Vector& r, double beta)
{
    p.eq(r,p,beta); // p = p*beta + r
}

template<>
inline void PCGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_alpha(const core::ExecParams* /*params*/ /* PARAMS FIRST */, Vector& x, Vector& r, Vector& p, Vector& q, double alpha)
{
#if 1 //SOFA_NO_VMULTIOP // unoptimized version
    x.peq(p,alpha);                 // x = x + alpha p
    r.peq(q,-alpha);                // r = r - alpha q
#else // single-operation optimization
    typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
    VMultiOp ops;
    ops.resize(2);
    ops[0].first = x;
    ops[0].second.push_back(std::make_pair((VecId)x,1.0));
    ops[0].second.push_back(std::make_pair((VecId)p,alpha));
    ops[1].first = (VecId)r;
    ops[1].second.push_back(std::make_pair((VecId)r,1.0));
    ops[1].second.push_back(std::make_pair((VecId)q,-alpha));
    simulation::tree::MechanicalVMultiOpVisitor vmop(ops);
    vmop.execute(this->getContext());
#endif
}

template<class TMatrix, class TVector>
void PCGLinearSolver<TMatrix,TVector>::solve (Matrix& M, Vector& x, Vector& b)
{
    using std::cerr;
    using std::endl;

    const core::ExecParams* params = core::ExecParams::defaultInstance();
    typename Inherit::TempVectorContainer vtmp(this, params, M, x, b);
    Vector& p = *vtmp.createTempVector();
    Vector& q = *vtmp.createTempVector();
    Vector& r = *vtmp.createTempVector();
    Vector& z = *vtmp.createTempVector();

    const bool printLog =  this->f_printLog.getValue();
    const bool verbose  = f_verbose.getValue();

    // -- solve the system using a conjugate gradient solution
    double rho, rho_1=0, alpha, beta;

    if( verbose ) cerr<<"PCGLinearSolver, b = "<< b <<endl;

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

    for( nb_iter=1; nb_iter<=f_maxIter.getValue(); nb_iter++ )
    {
        if (this->preconditioners.size()>0 && usePrecond)
        {
            sofa::helper::AdvancedTimer::stepBegin("PCGLinearSolver::apply Precond");

// 			for (unsigned int i=0;i<this->preconditioners.size();i++) {
// 				preconditioners[i]->setSystemLHVector(z);
// 				preconditioners[i]->setSystemRHVector(r);
// 				preconditioners[i]->solveSystem();
// 			}
            preconditioners[0]->updateSystemMatrix();
            preconditioners[0]->setSystemLHVector(z);
            preconditioners[0]->setSystemRHVector(r);
            preconditioners[0]->solveSystem();

            sofa::helper::AdvancedTimer::stepEnd("PCGLinearSolver::apply Precond");
        }
        else
        {
            z = r;
        }

        sofa::helper::AdvancedTimer::stepBegin("PCGLinearSolver::solve");

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
            cgstep_beta(params /* PARAMS FIRST */, p,z,beta);
        }

        if( verbose ) cerr<<"p : "<<p<<endl;

        // matrix-vector product
        q = M*p;

        if( verbose ) cerr<<"q = M p : "<<q<<endl;

        double den = p.dot(q);

        graph_den.push_back(den);

        if( fabs(den)<f_smallDenominatorThreshold.getValue() )
        {
            endcond = "threshold";
            if( verbose ) cerr<<"PCGLinearSolver, den = "<<den<<", smallDenominatorThreshold = "<<f_smallDenominatorThreshold.getValue()<<endl;
            break;
        }

        alpha = rho/den;
        //x.peq(p,alpha);                 // x = x + alpha p
        //r.peq(q,-alpha);                // r = r - alpha q
        cgstep_alpha(params /* PARAMS FIRST */, x,r,p,q,alpha);

        if( verbose )
        {
            cerr<<"den = "<<den<<", alpha = "<<alpha<<endl;
            cerr<<"x : "<<x<<endl;
            cerr<<"r : "<<r<<endl;
        }

        rho_1 = rho;

        sofa::helper::AdvancedTimer::stepEnd("PCGLinearSolver::solve");
    }
    sofa::helper::AdvancedTimer::stepEnd("PCGLinearSolver::solve");
    sofa::helper::AdvancedTimer::valSet("PCG iterations", nb_iter);

    f_graph.endEdit();
    // x is the solution of the system

    if( printLog ) cerr<<"PCGLinearSolver::solve, nbiter = "<<nb_iter<<" stop because of "<<endcond<<endl;

    if( verbose ) cerr<<"PCGLinearSolver::solve, solution = "<<x<<endl;

    vtmp.deleteTempVector(&p);
    vtmp.deleteTempVector(&q);
    vtmp.deleteTempVector(&r);
    vtmp.deleteTempVector(&z);
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

