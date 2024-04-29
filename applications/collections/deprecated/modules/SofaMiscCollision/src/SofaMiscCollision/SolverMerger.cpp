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
#include <SofaMiscCollision/SolverMerger.h>

#include <sofa/helper/FnDispatcher.h>
#include <sofa/helper/FnDispatcher.inl>

#include <sofa/component/odesolver/forward/EulerSolver.h>
#include <sofa/component/odesolver/forward/RungeKutta4Solver.h>
#include <sofa/component/odesolver/backward/StaticSolver.h>
#include <sofa/component/odesolver/backward/EulerImplicitSolver.h>
#include <sofa/component/linearsolver/iterative/CGLinearSolver.h>
#include <sofa/component/constraint/lagrangian/solver/LCPConstraintSolver.h>

namespace sofa::component::collision
{
using sofa::core::behavior::OdeSolver;
using sofa::core::behavior::BaseLinearSolver;
using sofa::core::behavior::ConstraintSolver;

SolverSet::SolverSet(core::behavior::OdeSolver::SPtr o,
                     core::behavior::BaseLinearSolver::SPtr l,
                     core::behavior::ConstraintSolver::SPtr c) :
        odeSolver(o),linearSolver(l),constraintSolver(c)
{}

namespace solvermergers
{

/// Create a new object which type is the template parameter, and
/// copy all its data fields values.
/// This function is meant to be used for ODE solvers and constraint solvers
template<class SolverType>
typename SolverType::SPtr copySolver(const SolverType& s)
{
    const SolverType* src = &s;
    typename SolverType::SPtr res = sofa::core::objectmodel::New<SolverType>();
    for (auto* dataField : src->getDataFields())
    {
        msg_error_when(dataField == nullptr, "SolverMerger::copySolver") << "Found nullptr data field from " << src->getName();
        if (auto* d = res->findData(dataField->getName()))
            d->copyValueFrom(dataField);
    }
    return res;
}

ConstraintSolver::SPtr createConstraintSolver(OdeSolver* solver1, OdeSolver* solver2)
{
    ConstraintSolver* csolver1 = nullptr;
    if (solver1!=nullptr)
    {
        solver1->getContext()->get(csolver1, core::objectmodel::BaseContext::SearchDown);
    }

    ConstraintSolver* csolver2 = nullptr;
    if (solver2!=nullptr)
    {
        solver2->getContext()->get(csolver2, core::objectmodel::BaseContext::SearchDown);
    }

    if (!csolver1 && !csolver2)
    {
        //no constraint solver associated to any ODE solver
        return nullptr;
    }
    if (!csolver1)
    {
        //first ODE solver does not have any constraint solver. The second is copied to be shared with the first
        if (auto* cs=dynamic_cast<constraint::lagrangian::solver::LCPConstraintSolver*>(csolver2))
            return copySolver<constraint::lagrangian::solver::LCPConstraintSolver>(*cs);
    }
    else if (!csolver2)
    {
        //second ODE solver does not have any constraint solver. The first is copied to be shared with the second
        if (auto* cs=dynamic_cast<constraint::lagrangian::solver::LCPConstraintSolver*>(csolver1))
            return copySolver<constraint::lagrangian::solver::LCPConstraintSolver>(*cs);
    }
    else
    {
        //both ODE solvers have an associated constraint solver
        if (auto* lcp1 = dynamic_cast<constraint::lagrangian::solver::LCPConstraintSolver*>(csolver1))
        if (auto* lcp2 = dynamic_cast<constraint::lagrangian::solver::LCPConstraintSolver*>(csolver2))
        {
            constraint::lagrangian::solver::LCPConstraintSolver::SPtr newSolver = sofa::core::objectmodel::New<constraint::lagrangian::solver::LCPConstraintSolver>();
            newSolver->d_initial_guess.setValue(lcp1->d_initial_guess.getValue() | lcp2->d_initial_guess.getValue());
            newSolver->d_build_lcp.setValue(lcp1->d_build_lcp.getValue() | lcp2->d_build_lcp.getValue());
            newSolver->d_tol.setValue(lcp1->d_tol.getValue() < lcp2->d_tol.getValue() ? lcp1->d_tol.getValue() : lcp2->d_tol.getValue() );
            newSolver->d_maxIt.setValue(lcp1->d_maxIt.getValue() > lcp2->d_maxIt.getValue() ? lcp1->d_maxIt.getValue() : lcp2->d_maxIt.getValue() );
            newSolver->d_mu.setValue((lcp1->d_mu.getValue() + lcp2->d_mu.getValue()) * 0.5);
            return newSolver;
        }
    }

    return nullptr;
}


// First the easy cases...

SolverSet createSolverEulerExplicitEulerExplicit(odesolver::forward::EulerExplicitSolver& solver1, odesolver::forward::EulerExplicitSolver& solver2)
{
    return SolverSet(copySolver<odesolver::forward::EulerExplicitSolver>(solver1), nullptr,createConstraintSolver(&solver1, &solver2));
}

SolverSet createSolverRungeKutta4RungeKutta4(odesolver::forward::RungeKutta4Solver& solver1, odesolver::forward::RungeKutta4Solver& solver2)
{
    return SolverSet(copySolver<odesolver::forward::RungeKutta4Solver>(solver1), nullptr,createConstraintSolver(&solver1, &solver2));
}

typedef linearsolver::iterative::CGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector> DefaultCGLinearSolver;

BaseLinearSolver::SPtr createLinearSolver(OdeSolver* solver1, OdeSolver* solver2)
{
    DefaultCGLinearSolver::SPtr lsolver = sofa::core::objectmodel::New<DefaultCGLinearSolver>();

    DefaultCGLinearSolver* lsolver1 = nullptr;
    if (solver1!=nullptr)
    {
        solver1->getContext()->get(lsolver1, core::objectmodel::BaseContext::SearchDown);
    }

    DefaultCGLinearSolver* lsolver2 = nullptr;
    if (solver2!=nullptr)
    {
        solver2->getContext()->get(lsolver2, core::objectmodel::BaseContext::SearchDown);
    }

    unsigned int maxIter = 0;
    double tolerance = 1.0e10;
    double smallDenominatorThreshold = 1.0e10;
    if (lsolver1)
    {
        if (lsolver1->d_maxIter.getValue() > maxIter) maxIter = lsolver1->d_maxIter.getValue();
        if (lsolver1->d_tolerance.getValue() < tolerance) tolerance = lsolver1->d_tolerance.getValue();
        if (lsolver1->d_smallDenominatorThreshold.getValue() < smallDenominatorThreshold) smallDenominatorThreshold = lsolver1->d_smallDenominatorThreshold.getValue();
    }
    if (lsolver2)
    {
        if (lsolver2->d_maxIter.getValue() > maxIter) maxIter = lsolver2->d_maxIter.getValue();
        if (lsolver2->d_tolerance.getValue() < tolerance) tolerance = lsolver2->d_tolerance.getValue();
        if (lsolver2->d_smallDenominatorThreshold.getValue() < smallDenominatorThreshold) smallDenominatorThreshold = lsolver2->d_smallDenominatorThreshold.getValue();
    }
    lsolver->d_maxIter.setValue( maxIter );
    lsolver->d_tolerance.setValue( tolerance );
    lsolver->d_smallDenominatorThreshold.setValue( smallDenominatorThreshold );
    return lsolver;
}

SolverSet createSolverEulerImplicitEulerImplicit(odesolver::backward::EulerImplicitSolver& solver1, odesolver::backward::EulerImplicitSolver& solver2)
{
    odesolver::backward::EulerImplicitSolver::SPtr solver = sofa::core::objectmodel::New<odesolver::backward::EulerImplicitSolver>();
    solver->f_rayleighStiffness.setValue( solver1.f_rayleighStiffness.getValue() < solver2.f_rayleighStiffness.getValue() ? solver1.f_rayleighStiffness.getValue() : solver2.f_rayleighStiffness.getValue() );
    solver->f_rayleighMass.setValue( solver1.f_rayleighMass.getValue() < solver2.f_rayleighMass.getValue() ? solver1.f_rayleighMass.getValue() : solver2.f_rayleighMass.getValue() );
    solver->f_velocityDamping.setValue( solver1.f_velocityDamping.getValue() > solver2.f_velocityDamping.getValue() ? solver1.f_velocityDamping.getValue() : solver2.f_velocityDamping.getValue());
    return SolverSet(solver,
            createLinearSolver(&solver1, &solver2),
            createConstraintSolver(&solver1, &solver2));
}

SolverSet createSolverStaticSolver(odesolver::backward::StaticSolver& solver1, odesolver::backward::StaticSolver& solver2)
{
    return SolverSet(copySolver<odesolver::backward::StaticSolver>(solver1),
            createLinearSolver(&solver1, &solver2),
            createConstraintSolver(&solver1, &solver2));
}

// Then the other, with the policy of taking the more precise solver

SolverSet createSolverRungeKutta4Euler(odesolver::forward::RungeKutta4Solver& solver1, odesolver::forward::EulerExplicitSolver& solver2)
{
    return SolverSet(copySolver<odesolver::forward::RungeKutta4Solver>(solver1), nullptr,createConstraintSolver(&solver1, &solver2));
}

SolverSet createSolverEulerImplicitEuler(odesolver::backward::EulerImplicitSolver& solver1, odesolver::forward::EulerExplicitSolver& solver2)
{
    return SolverSet(copySolver<odesolver::backward::EulerImplicitSolver>(solver1),
            createLinearSolver(&solver1, nullptr),
            createConstraintSolver(&solver1, &solver2));
}

SolverSet createSolverEulerImplicitRungeKutta4(odesolver::backward::EulerImplicitSolver& solver1, odesolver::forward::RungeKutta4Solver& solver2)
{
    return SolverSet(copySolver<odesolver::backward::EulerImplicitSolver>(solver1),
            createLinearSolver(&solver1, nullptr),
            createConstraintSolver(&solver1, &solver2));
}

}// namespace SolverMergers

using namespace solvermergers;


SolverMerger* SolverMerger::getInstance()
{
    static SolverMerger instance;
    return &instance;
}

SolverSet SolverMerger::merge(core::behavior::OdeSolver* solver1, core::behavior::OdeSolver* solver2)
{
    return getInstance()->solverDispatcher.go(*solver1, *solver2);
}

SolverMerger::SolverMerger()
{
    solverDispatcher.add<odesolver::forward::EulerExplicitSolver,odesolver::forward::EulerExplicitSolver,createSolverEulerExplicitEulerExplicit,false>();
    solverDispatcher.add<odesolver::forward::RungeKutta4Solver,odesolver::forward::RungeKutta4Solver,createSolverRungeKutta4RungeKutta4,false>();
    solverDispatcher.add<odesolver::backward::EulerImplicitSolver,odesolver::backward::EulerImplicitSolver,createSolverEulerImplicitEulerImplicit,false>();
    solverDispatcher.add<odesolver::forward::RungeKutta4Solver,odesolver::forward::EulerExplicitSolver,createSolverRungeKutta4Euler,true>();
    solverDispatcher.add<odesolver::backward::EulerImplicitSolver,odesolver::forward::EulerExplicitSolver,createSolverEulerImplicitEuler,true>();
    solverDispatcher.add<odesolver::backward::EulerImplicitSolver,odesolver::forward::RungeKutta4Solver,createSolverEulerImplicitRungeKutta4,true>();
    solverDispatcher.add<odesolver::backward::StaticSolver,odesolver::backward::StaticSolver,createSolverStaticSolver,true>();
}

} // namespace sofa::component::collision
