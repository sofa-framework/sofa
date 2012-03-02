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
#include <sofa/component/linearsolver/ShewchukPCGLinearSolver.h>
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
ShewchukPCGLinearSolver<TMatrix,TVector>::ShewchukPCGLinearSolver()
    : f_maxIter( initData(&f_maxIter,(unsigned)25,"iterations","maximum number of iterations of the Conjugate Gradient solution") )
    , f_tolerance( initData(&f_tolerance,1e-5,"tolerance","desired precision of the Conjugate Gradient Solution (ratio of current residual norm over initial residual norm)") )
    , f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , f_update_iteration( initData(&f_update_iteration,(unsigned)0,"update_iteration","Number of CG iterations before next refresh of precondtioner") )
    , f_update_step( initData(&f_update_step,(unsigned)1,"update_step","Number of steps before the next refresh of precondtioners") )
    , f_use_precond( initData(&f_use_precond,true,"use_precond","Use preconditioner") )
    , f_build_precond( initData(&f_build_precond,true,"build_precond","Build the preconditioners, if false build the preconditioner only at the initial step") )
    , f_use_first_precond( initData(&f_use_first_precond,false,"use_first_precond","Use only first precond") )
    , f_preconditioners( initData(&f_preconditioners, "preconditioners", "If not empty: path to the solvers to use as preconditioners") )
    , f_graph( initData(&f_graph,"graph","Graph of residuals at each iteration") )
{
    f_graph.setWidget("graph");
//    f_graph.setReadOnly(true);
    first = true;
}

template<class TMatrix, class TVector>
void ShewchukPCGLinearSolver<TMatrix,TVector>::init()
{
    std::vector<sofa::core::behavior::LinearSolver*> solvers;
    BaseContext * c = this->getContext();

    const helper::vector<std::string>& precondNames = f_preconditioners.getValue();
    if (precondNames.empty())
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
void ShewchukPCGLinearSolver<TMatrix,TVector>::setSystemMBKMatrix(const core::MechanicalParams* mparams)
{
    sofa::helper::AdvancedTimer::valSet("PCG::buildMBK", 1);
    sofa::helper::AdvancedTimer::stepBegin("PCG::setSystemMBKMatrix");

    Inherit::setSystemMBKMatrix(mparams);

    sofa::helper::AdvancedTimer::stepEnd("PCG::setSystemMBKMatrix(Precond)");

    if (preconditioners.size()==0) return;

    if (first)   //We initialize all the preconditioners for the first step
    {
        for (unsigned int i=0; i<preconditioners.size(); ++i)
        {
            preconditioners[i]->setSystemMBKMatrix(mparams);
        }
        first = false;
        next_refresh_step = 1;
    }
    else if (f_build_precond.getValue())     // We use only the first precond in the list
    {
        sofa::helper::AdvancedTimer::valSet("PCG::PrecondBuildMBK", 1);
        sofa::helper::AdvancedTimer::stepBegin("PCG::PrecondSetSystemMBKMatrix");

        if ((f_update_step.getValue()>0) && (f_update_iteration.getValue()>0))
        {
            if ((next_refresh_step>=f_update_step.getValue()) && (next_refresh_iteration>=f_update_iteration.getValue()))
            {
                for (unsigned int i=0; i<preconditioners.size(); ++i)
                {
                    preconditioners[i]->setSystemMBKMatrix(mparams);
                }
                next_refresh_step=1;
            }
            else
            {
                for (unsigned int i=0; i<preconditioners.size(); ++i)
                {
                    preconditioners[i]->updateSystemMatrix();
                }
                next_refresh_step++;
            }
        }
        else if (f_update_step.getValue()>0)
        {
            if (next_refresh_step>=f_update_step.getValue())
            {
                for (unsigned int i=0; i<preconditioners.size(); ++i)
                {
                    preconditioners[i]->setSystemMBKMatrix(mparams);
                }
                next_refresh_step=1;
            }
            else
            {
                for (unsigned int i=0; i<preconditioners.size(); ++i)
                {
                    preconditioners[i]->updateSystemMatrix();
                }
                next_refresh_step++;
            }
        }
        else if (f_update_iteration.getValue()>0)
        {
            if (next_refresh_iteration>=f_update_iteration.getValue())
            {
                for (unsigned int i=0; i<preconditioners.size(); ++i)
                {
                    preconditioners[i]->setSystemMBKMatrix(mparams);
                }
                next_refresh_iteration=1;
            }
            else
            {
                for (unsigned int i=0; i<preconditioners.size(); ++i)
                {
                    preconditioners[i]->updateSystemMatrix();
                }
            }
        }
        sofa::helper::AdvancedTimer::stepEnd("PCG::PrecondSetSystemMBKMatrix");
    }
    next_refresh_iteration = 1;
}

template<>
inline void ShewchukPCGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_beta(Vector& p, Vector& r, double beta)
{
    p.eq(r,p,beta); // p = p*beta + r
}

template<>
inline void ShewchukPCGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_alpha(Vector& x, Vector& p, double alpha)
{
    x.peq(p,alpha);                 // x = x + alpha p
}

template<class TMatrix, class TVector>
void ShewchukPCGLinearSolver<TMatrix,TVector>::solve (Matrix& M, Vector& x, Vector& b)
{
    sofa::helper::AdvancedTimer::stepBegin("PCGLinearSolver::solve");
    const core::ExecParams* params = core::ExecParams::defaultInstance();
    typename Inherit::TempVectorContainer vtmp(this, params, M, x, b);
    Vector& r = *vtmp.createTempVector();
    Vector& d = *vtmp.createTempVector();
    Vector& q = *vtmp.createTempVector();
    Vector& s = *vtmp.createTempVector();
    std::map < std::string, sofa::helper::vector<double> >& graph = * f_graph.beginEdit();
    sofa::helper::vector<double>& graph_error = graph["Error"];
    graph_error.clear();

    unsigned iter=1;

    bool apply_precond = false;
    if ((this->preconditioners.size()>0) && f_build_precond.getValue()) apply_precond = f_use_precond.getValue();
    else apply_precond = false;

    if (apply_precond)
    {
        sofa::helper::AdvancedTimer::stepEnd("PCGLinearSolver::solve");
        sofa::helper::AdvancedTimer::stepBegin("PCGLinearSolver::apply Precond");
        preconditioners[0]->setSystemLHVector(d);
        preconditioners[0]->setSystemRHVector(b);
        preconditioners[0]->solveSystem();

        if ((preconditioners.size() > 1) && (!f_use_first_precond.getValue()))   // use if multiple preconds
        {
            Vector& t = *vtmp.createTempVector();
            for (unsigned int i=1; i<preconditioners.size(); ++i)
            {
                t = d;
                preconditioners[i]->setSystemLHVector(d);
                preconditioners[i]->setSystemRHVector(t);
                preconditioners[i]->solveSystem();
            }
            vtmp.deleteTempVector(&t);
        }
        sofa::helper::AdvancedTimer::stepEnd("PCGLinearSolver::apply Precond");
        sofa::helper::AdvancedTimer::stepBegin("PCGLinearSolver::solve");
    }
    else
    {
        d = b;
    }

    x.clear();
    r = b;
    double deltaNew = b.dot(d);
    double delta0 = deltaNew;
    double eps = f_tolerance.getValue() * f_tolerance.getValue() * delta0;

    while ((iter <= f_maxIter.getValue()) && (deltaNew > eps))
    {
        graph_error.push_back(sqrt(deltaNew));

        q = M * d;
        double dtq = d.dot(q);
        double alpha = deltaNew / dtq;

        cgstep_alpha(x,d,alpha);//for(int i=0; i<n; i++) x[i] += alpha * d[i];
        cgstep_alpha(r,q,-alpha);//for (int i=0; i<n; i++) r[i] = r[i] - alpha * q[i];

        if (this->preconditioners.size()>0 && f_build_precond.getValue()) apply_precond = f_use_precond.getValue();
        else apply_precond = false;

        if (apply_precond)
        {
            sofa::helper::AdvancedTimer::stepEnd("PCGLinearSolver::solve");
            sofa::helper::AdvancedTimer::stepBegin("PCGLinearSolver::apply Precond");
            preconditioners[0]->setSystemLHVector(s);
            preconditioners[0]->setSystemRHVector(r);
            preconditioners[0]->solveSystem();

            if ((preconditioners.size()>1) && (!f_use_first_precond.getValue()))  // use if multiple preconds
            {
                Vector& t = *vtmp.createTempVector();
                for (unsigned int i=1; i<preconditioners.size(); ++i)
                {
                    t = s;
                    preconditioners[i]->setSystemLHVector(s);
                    preconditioners[i]->setSystemRHVector(t);
                    preconditioners[i]->solveSystem();
                }
                vtmp.deleteTempVector(&t);
            }

            sofa::helper::AdvancedTimer::stepEnd("PCGLinearSolver::apply Precond");
            sofa::helper::AdvancedTimer::stepBegin("PCGLinearSolver::solve");
        }
        else
        {
            s = r;
        }


        double deltaOld = deltaNew;
        deltaNew = r.dot(s);
        double beta = deltaNew / deltaOld;

        cgstep_beta(d,s,beta);//for (int i=0; i<n; i++) d[i] = r[i] + beta * d[i];

        iter++;
    }

    graph_error.push_back(sqrt(deltaNew));
    next_refresh_iteration=iter;
    sofa::helper::AdvancedTimer::valSet("PCG iterations", iter);

    f_graph.endEdit();
    vtmp.deleteTempVector(&r);
    vtmp.deleteTempVector(&q);
    vtmp.deleteTempVector(&d);
    vtmp.deleteTempVector(&s);
    sofa::helper::AdvancedTimer::stepEnd("PCGLinearSolver::solve");
}

SOFA_DECL_CLASS(ShewchukPCGLinearSolver)

int ShewchukPCGLinearSolverClass = core::RegisterObject("Linear system solver using the conjugate gradient iterative algorithm")
        .add< ShewchukPCGLinearSolver<GraphScatteredMatrix,GraphScatteredVector> >(true)
        ;

} // namespace linearsolver

} // namespace component

} // namespace sofa

