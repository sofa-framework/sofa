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
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>
#include <sofa/component/linearsolver/iterative/PCGLinearSolver.h>
#include <sofa/component/linearsolver/iterative/PreconditionedMatrixFreeSystem.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/helper/map.h>
#include <sofa/simulation/AnimateBeginEvent.h>

#include <cmath>

namespace sofa::component::linearsolver::iterative
{

template<class TMatrix, class TVector>
PCGLinearSolver<TMatrix,TVector>::PCGLinearSolver()
    : d_maxIter(initData(&d_maxIter, 25u, "iterations", "Maximum number of iterations after which the iterative descent of the Conjugate Gradient must stop") )
    , d_tolerance(initData(&d_tolerance, 1e-5, "tolerance", "Desired accuracy of the Conjugate Gradient solution evaluating: |r|²/|b|² (ratio of current residual norm over initial residual norm)") )
    , d_use_precond(initData(&d_use_precond, true, "use_precond", "Use a preconditioner") )
    , l_preconditioner(initLink("preconditioner", "Link towards the linear solver used to precondition the conjugate gradient"))
    , d_update_step(this, "v25.12", "v26.06", "update_step", "Instead, use the Data 'assemblingRate' in the associated PreconditionedMatrixFreeSystem")
    , d_graph(initData(&d_graph, "graph", "Graph of residuals at each iteration") )
    , next_refresh_step(0)
    , newton_iter(0)
{
    d_graph.setWidget("graph");
    first = true;
    this->f_listening.setValue(true);
}

template <class TMatrix, class TVector>
void PCGLinearSolver<TMatrix, TVector>::init()
{
    Inherit1::init();

    // Find linear solvers
    if (l_preconditioner.empty())
    {
        msg_info() << "Link '" << l_preconditioner.getName() << "' to the desired linear solver "
            "should be set to precondition the conjugate gradient. Without preconditioner, the "
            "solver will act as a regular conjugate gradient solver.";
    }
    else
    {
        if (l_preconditioner.get() == nullptr)
        {
            msg_error() << "No preconditioner found at path: " << l_preconditioner.getLinkedPath();
            this->d_componentState.setValue( sofa::core::objectmodel::ComponentState::Invalid);
            return;
        }
        else
        {
            if (l_preconditioner->getTemplateName() == "GraphScattered")
            {
                msg_error() << "Cannot use the preconditioner " << l_preconditioner->getName()
                            << " because it is templated on GraphScatteredType";
                this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
                return;
            }
            else
            {
                msg_info() << "Preconditioner path used: '" << l_preconditioner.getLinkedPath() << "'";
            }
        }
    }

    ensureRequiredLinearSystemType();
    if (this->isComponentStateInvalid())
        return;

    first = true;

    if (!this->isComponentStateInvalid())
    {
        this->d_componentState.setValue( sofa::core::objectmodel::ComponentState::Valid);
    }
}

template <class TMatrix, class TVector>
void PCGLinearSolver<TMatrix, TVector>::ensureRequiredLinearSystemType()
{
    if (this->l_linearSystem)
    {
        auto* preconditionedMatrix =
            dynamic_cast<PreconditionedMatrixFreeSystem<TMatrix, TVector>*>(this->l_linearSystem.get());
        if (!preconditionedMatrix)
        {
            msg_error() << "This linear solver is designed to work with a "
                        << PreconditionedMatrixFreeSystem<TMatrix, TVector>::GetClass()->className
                        << " linear system, but a " << this->l_linearSystem->getClassName()
                        << " was found";
            this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        }
    }
}

template <class TMatrix, class TVector>
void PCGLinearSolver<TMatrix, TVector>::bwdInit()
{
    if (this->isComponentStateInvalid())
        return;

    //link the linear systems of both the preconditioner and the PCG
    if (l_preconditioner && this->l_linearSystem)
    {
        if (auto* preconditionerLinearSystem = l_preconditioner->getLinearSystem())
        {
            if (auto* preconditionedMatrix = dynamic_cast<PreconditionedMatrixFreeSystem<TMatrix,TVector>*>(this->l_linearSystem.get()))
            {
                msg_info() << "Linking the preconditioner linear system (" << preconditionerLinearSystem->getPathName() << ") to the PCG linear system (" << preconditionedMatrix->getPathName() << ")";
                //this link is essential to ensure that the preconditioner matrix is assembled
                preconditionedMatrix->l_preconditionerSystem.set(preconditionerLinearSystem);
            }
            else
            {
                msg_error() << "The preconditioned linear system (" << preconditionedMatrix->getPathName() << ") is not a PreconditionedMatrixFreeSystem";
                this->d_componentState.setValue( sofa::core::objectmodel::ComponentState::Invalid);
                return;
            }
        }
    }
}

template <>
inline void PCGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_beta(Vector& p, Vector& r, Real beta)
{
    p.eq(r,p,beta); // p = p*beta + r
}

template<>
inline void PCGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_alpha(Vector& x, Vector& p, Real alpha)
{
    x.peq(p,alpha);                 // x = x + alpha p
}

template <class Matrix, class Vector>
void PCGLinearSolver<Matrix, Vector>::handleEvent(sofa::core::objectmodel::Event* event)
{
    /// this event shoul be launch before the addKToMatrix
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event))
    {
        newton_iter = 0;
        std::map<std::string, sofa::type::vector<Real> >& graph = *d_graph.beginEdit();
        graph.clear();
    }
}

template <class TMatrix, class TVector>
void PCGLinearSolver<TMatrix, TVector>::checkLinearSystem()
{
    // a PreconditionedMatrixFreeSystem component is created in the absence of a linear system
    this->template doCheckLinearSystem<PreconditionedMatrixFreeSystem<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector> >();
}

template <class TMatrix, class TVector>
void PCGLinearSolver<TMatrix,TVector>::solve (Matrix& M, Vector& x, Vector& b)
{
    SCOPED_TIMER_VARNAME(solveTimer, "PCGLinearSolver::solve");

    std::map < std::string, sofa::type::vector<Real> >& graph = * d_graph.beginEdit();
//    sofa::type::vector<Real>& graph_error = graph["Error"];

    newton_iter++;
    char name[256];
    sprintf(name,"Error %d",newton_iter);
    sofa::type::vector<Real>& graph_error = graph[std::string(name)];

    const core::ExecParams* params = core::execparams::defaultInstance();
    typename Inherit::TempVectorContainer vtmp(this, params, M, x, b);
    Vector& r = *vtmp.createTempVector();
    Vector& w = *vtmp.createTempVector();
    Vector& s = *vtmp.createTempVector();

    const bool apply_precond = l_preconditioner.get()!=nullptr && d_use_precond.getValue();

    const Real b_norm = b.dot(b);
    const Real tol = d_tolerance.getValue() * b_norm;

    r = M * x;
    cgstep_beta(r,b,-1);// r = -1 * r + b  =   b - (M * x)

    if (apply_precond)
    {
        SCOPED_TIMER_VARNAME(applyPrecondTimer, "PCGLinearSolver::apply Precond");
        l_preconditioner->getLinearSystem()->setSystemSolution(w);
        l_preconditioner->getLinearSystem()->setRHS(r);
        l_preconditioner->solveSystem();
        l_preconditioner->getLinearSystem()->dispatchSystemSolution(w);
    }
    else
    {
        w = r;
    }

    Real r_norm = r.dot(w);
    graph_error.push_back(r_norm/b_norm);

    unsigned iter=1;
    while ((iter <= d_maxIter.getValue()) && (r_norm > tol))
    {
        s = M * w;
        const Real dtq = w.dot(s);
        Real alpha = r_norm / dtq;

        cgstep_alpha(x,w,alpha);//for(int i=0; i<n; i++) x[i] += alpha * d[i];
        cgstep_alpha(r,s,-alpha);//for (int i=0; i<n; i++) r[i] = r[i] - alpha * q[i];

        if (apply_precond)
        {
            SCOPED_TIMER_VARNAME(applyPrecondTimer, "PCGLinearSolver::apply Precond");
            l_preconditioner->getLinearSystem()->setSystemSolution(s);
            l_preconditioner->getLinearSystem()->setRHS(r);
            l_preconditioner->solveSystem();
            l_preconditioner->getLinearSystem()->dispatchSystemSolution(s);
        }
        else
        {
            s = r;
        }

        const Real deltaOld = r_norm;
        r_norm = r.dot(s);
        graph_error.push_back(r_norm/b_norm);

        Real beta = r_norm / deltaOld;

        cgstep_beta(w,s,beta);//for (int i=0; i<n; i++) d[i] = r[i] + beta * d[i];

        iter++;
    }

    d_graph.endEdit();

    vtmp.deleteTempVector(&r);
    vtmp.deleteTempVector(&w);
    vtmp.deleteTempVector(&s);

    sofa::helper::AdvancedTimer::valSet("PCG iterations", iter);
}

} // namespace sofa::component::linearsolver::iterative
