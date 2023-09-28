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
#include <sofa/component/linearsolver/iterative/ShewchukPCGLinearSolver.h>

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/helper/map.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

#include <cmath>

namespace sofa::component::linearsolver::iterative
{

template<class TMatrix, class TVector>
ShewchukPCGLinearSolver<TMatrix,TVector>::ShewchukPCGLinearSolver()
    : f_maxIter( initData(&f_maxIter,(unsigned)25,"iterations","maximum number of iterations of the Conjugate Gradient solution") )
    , f_tolerance( initData(&f_tolerance,1e-5,"tolerance","desired precision of the Conjugate Gradient Solution (ratio of current residual norm over initial residual norm)") )
    , f_use_precond( initData(&f_use_precond,true,"use_precond","Use preconditioner") )
    , l_preconditioner(initLink("preconditioner", "Link towards the linear solver used to precondition the conjugate gradient"))
    , f_update_step( initData(&f_update_step,(unsigned)1,"update_step","Number of steps before the next refresh of precondtioners") )
    , f_build_precond( initData(&f_build_precond,true,"build_precond","Build the preconditioners, if false build the preconditioner only at the initial step") )
    , f_graph( initData(&f_graph,"graph","Graph of residuals at each iteration") )
    , next_refresh_step(0)
    , newton_iter(0)
{
    f_graph.setWidget("graph");
    first = true;
    this->f_listening.setValue(true);
}

template<class TMatrix, class TVector>
void ShewchukPCGLinearSolver<TMatrix,TVector>::init()
{
    Inherit1::init();

    // Find linear solvers
    if (l_preconditioner.empty())
    {
        msg_info() << "Link \"preconditioner\" to the desired linear solver should be set to precondition the conjugate gradient.";
    }
    else
    {
        if (l_preconditioner.get() == nullptr)
        {
            msg_error() << "No preconditioner found at path: " << l_preconditioner.getLinkedPath();
            sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            return;
        }
        else
        {
            if (l_preconditioner.get()->getTemplateName() == "GraphScattered")
            {
                msg_error() << "Can not use the preconditioner " << l_preconditioner.get()->getName() << " because it is templated on GraphScatteredType";
                sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
                return;
            }
            else
            {
                msg_info() << "Preconditioner path used: '" << l_preconditioner.getLinkedPath() << "'";
            }
        }
    }

    first = true;
    sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

template<class TMatrix, class TVector>
void ShewchukPCGLinearSolver<TMatrix,TVector>::setSystemMBKMatrix(const core::MechanicalParams* mparams)
{
    sofa::helper::AdvancedTimer::valSet("PCG::buildMBK", 1);

    {
        SCOPED_TIMER("PCG::setSystemMBKMatrix");
        Inherit::setSystemMBKMatrix(mparams);
    }

    if (l_preconditioner.get()==nullptr) return;

    if (first) //We initialize all the preconditioners for the first step
    {
        l_preconditioner.get()->setSystemMBKMatrix(mparams);
        first = false;
        next_refresh_step = 1;
    }
    else if (f_build_precond.getValue())
    {
        sofa::helper::AdvancedTimer::valSet("PCG::PrecondBuildMBK", 1);
        SCOPED_TIMER_VARNAME(mbkTimer, "PCG::PrecondSetSystemMBKMatrix");

        if (f_update_step.getValue()>0)
        {
            if (next_refresh_step>=f_update_step.getValue())
            {
                l_preconditioner.get()->setSystemMBKMatrix(mparams);
                next_refresh_step=1;
            }
            else
            {
                next_refresh_step++;
            }
        }
    }

    l_preconditioner.get()->updateSystemMatrix();
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

template<class Matrix, class Vector>
void ShewchukPCGLinearSolver<Matrix,Vector>::handleEvent(sofa::core::objectmodel::Event* event) {
    /// this event shoul be launch before the addKToMatrix
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event))
    {
        newton_iter = 0;
        std::map < std::string, sofa::type::vector<double> >& graph = * f_graph.beginEdit();
        graph.clear();
    }
}


template<class TMatrix, class TVector>
void ShewchukPCGLinearSolver<TMatrix,TVector>::solve (Matrix& M, Vector& x, Vector& b)
{
    SCOPED_TIMER_VARNAME(solveTimer, "PCGLinearSolver::solve");

    std::map < std::string, sofa::type::vector<double> >& graph = * f_graph.beginEdit();
//    sofa::type::vector<double>& graph_error = graph["Error"];

    newton_iter++;
    char name[256];
    sprintf(name,"Error %d",newton_iter);
    sofa::type::vector<double>& graph_error = graph[std::string(name)];

    const core::ExecParams* params = core::execparams::defaultInstance();
    typename Inherit::TempVectorContainer vtmp(this, params, M, x, b);
    Vector& r = *vtmp.createTempVector();
    Vector& w = *vtmp.createTempVector();
    Vector& s = *vtmp.createTempVector();

    const bool apply_precond = l_preconditioner.get()!=nullptr && f_use_precond.getValue();

    const double b_norm = b.dot(b);
    const double tol = f_tolerance.getValue() * b_norm;

    r = M * x;
    cgstep_beta(r,b,-1);// r = -1 * r + b  =   b - (M * x)

    if (apply_precond)
    {
        SCOPED_TIMER_VARNAME(applyPrecondTimer, "PCGLinearSolver::apply Precond");
        l_preconditioner.get()->setSystemLHVector(w);
        l_preconditioner.get()->setSystemRHVector(r);
        l_preconditioner.get()->solveSystem();
    }
    else
    {
        w = r;
    }

    double r_norm = r.dot(w);
    graph_error.push_back(r_norm/b_norm);

    unsigned iter=1;
    while ((iter <= f_maxIter.getValue()) && (r_norm > tol))
    {
        s = M * w;
        const double dtq = w.dot(s);
        double alpha = r_norm / dtq;

        cgstep_alpha(x,w,alpha);//for(int i=0; i<n; i++) x[i] += alpha * d[i];
        cgstep_alpha(r,s,-alpha);//for (int i=0; i<n; i++) r[i] = r[i] - alpha * q[i];

        if (apply_precond)
        {
            SCOPED_TIMER_VARNAME(applyPrecondTimer, "PCGLinearSolver::apply Precond");
            l_preconditioner.get()->setSystemLHVector(s);
            l_preconditioner.get()->setSystemRHVector(r);
            l_preconditioner.get()->solveSystem();
        }
        else
        {
            s = r;
        }

        const double deltaOld = r_norm;
        r_norm = r.dot(s);
        graph_error.push_back(r_norm/b_norm);

        double beta = r_norm / deltaOld;

        cgstep_beta(w,s,beta);//for (int i=0; i<n; i++) d[i] = r[i] + beta * d[i];

        iter++;
    }

    f_graph.endEdit();

    vtmp.deleteTempVector(&r);
    vtmp.deleteTempVector(&w);
    vtmp.deleteTempVector(&s);

    sofa::helper::AdvancedTimer::valSet("PCG iterations", iter);
}

} // namespace sofa::component::linearsolver::iterative
