/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/collision/SolverMerger.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/helper/FnDispatcher.h>
#include <sofa/helper/FnDispatcher.inl>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/component/odesolver/RungeKutta4Solver.h>
#include <sofa/component/odesolver/StaticSolver.h>
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>
#include <sofa/component/constraintset/LCPConstraintSolver.h>
#ifdef SOFA_HAVE_EIGEN2
#include <sofa/component/constraintset/LMConstraintSolver.h>
#endif
namespace sofa
{

namespace component
{

namespace collision
{
using sofa::core::behavior::OdeSolver;
using sofa::core::behavior::LinearSolver;
using sofa::core::behavior::ConstraintSolver;

namespace solvermergers
{

template<class T>
typename T::SPtr copySolver(const T& s)
{
    const T* src = &s;
    typename T::SPtr res = sofa::core::objectmodel::New<T>();
    const sofa::core::objectmodel::BaseObject::VecData& fields = src->getDataFields();
    for (unsigned int i=0; i<fields.size(); ++i)
    {
        core::objectmodel::BaseData* s = fields[i];
        core::objectmodel::BaseData* d = res->findData(s->getName());
        if (d)
            d->copyValue(s);
    }
    return res;
}

ConstraintSolver::SPtr createConstraintSolver(OdeSolver* solver1, OdeSolver* solver2)
{
    ConstraintSolver* csolver1 = NULL; if (solver1!=NULL) solver1->getContext()->get(csolver1, core::objectmodel::BaseContext::SearchDown);
    ConstraintSolver* csolver2 = NULL; if (solver2!=NULL) solver2->getContext()->get(csolver2, core::objectmodel::BaseContext::SearchDown);

    if (!csolver1 && !csolver2) return NULL;
    if (!csolver1)
    {
        if (constraintset::LCPConstraintSolver* cs=dynamic_cast<constraintset::LCPConstraintSolver*>(csolver2))
            return copySolver<constraintset::LCPConstraintSolver>(*cs);
#ifdef SOFA_HAVE_EIGEN2
        else if (constraintset::LMConstraintSolver* cs=dynamic_cast<constraintset::LMConstraintSolver*>(csolver2))
            return copySolver<constraintset::LMConstraintSolver>(*cs);
#endif
    }
    else if (!csolver2)
    {
        if (constraintset::LCPConstraintSolver* cs=dynamic_cast<constraintset::LCPConstraintSolver*>(csolver1))
            return copySolver<constraintset::LCPConstraintSolver>(*cs);
#ifdef SOFA_HAVE_EIGEN2
        else if (constraintset::LMConstraintSolver* cs=dynamic_cast<constraintset::LMConstraintSolver*>(csolver1))
            return copySolver<constraintset::LMConstraintSolver>(*cs);
#endif
    }
    else
    {
        if (dynamic_cast<constraintset::LCPConstraintSolver*>(csolver2) && dynamic_cast<constraintset::LCPConstraintSolver*>(csolver1))
        {
            constraintset::LCPConstraintSolver* lcp1=dynamic_cast<constraintset::LCPConstraintSolver*>(csolver1);
            constraintset::LCPConstraintSolver* lcp2=dynamic_cast<constraintset::LCPConstraintSolver*>(csolver2);
            constraintset::LCPConstraintSolver::SPtr newSolver = sofa::core::objectmodel::New<constraintset::LCPConstraintSolver>();
            newSolver->displayTime.setValue(lcp1->displayTime.getValue() | lcp2->displayTime.getValue());
            newSolver->initial_guess.setValue(lcp1->initial_guess.getValue() | lcp2->initial_guess.getValue());
            newSolver->build_lcp.setValue(lcp1->build_lcp.getValue() | lcp2->build_lcp.getValue());
            newSolver->tol.setValue(lcp1->tol.getValue() < lcp2->tol.getValue() ? lcp1->tol.getValue() : lcp2->tol.getValue() );
            newSolver->maxIt.setValue(lcp1->maxIt.getValue() > lcp2->maxIt.getValue() ? lcp1->maxIt.getValue() : lcp2->maxIt.getValue() );
            newSolver->mu.setValue((lcp1->mu.getValue() + lcp2->mu.getValue())*0.5);
            return newSolver;
        }
#ifdef SOFA_HAVE_EIGEN2
        else if (dynamic_cast<constraintset::LMConstraintSolver*>(csolver2) && dynamic_cast<constraintset::LMConstraintSolver*>(csolver1))
        {
            constraintset::LMConstraintSolver* lm1=dynamic_cast<constraintset::LMConstraintSolver*>(csolver1);
            constraintset::LMConstraintSolver* lm2=dynamic_cast<constraintset::LMConstraintSolver*>(csolver2);
            constraintset::LMConstraintSolver::SPtr newSolver = sofa::core::objectmodel::New<constraintset::LMConstraintSolver>();
            newSolver->numIterations.setValue(lm1->numIterations.getValue() > lm2->numIterations.getValue() ? lm1->numIterations.getValue() : lm2->numIterations.getValue() );
            newSolver->maxError.setValue(lm1->maxError.getValue() < lm2->maxError.getValue() ? lm1->maxError.getValue() : lm2->maxError.getValue() );

            newSolver->constraintAcc.setValue(lm1->constraintAcc.getValue() | lm2->constraintAcc.getValue());
            newSolver->constraintVel.setValue(lm1->constraintVel.getValue() | lm2->constraintVel.getValue());
            newSolver->constraintPos.setValue(lm1->constraintPos.getValue() | lm2->constraintPos.getValue());
            return newSolver;
        }
#endif
    }

    return NULL;
}


// First the easy cases...

SolverSet createSolverEulerEuler(odesolver::EulerSolver& solver1, odesolver::EulerSolver& solver2)
{
    return SolverSet(copySolver<odesolver::EulerSolver>(solver1), NULL,createConstraintSolver(&solver1, &solver2));
}

SolverSet createSolverRungeKutta4RungeKutta4(odesolver::RungeKutta4Solver& solver1, odesolver::RungeKutta4Solver& solver2)
{
    return SolverSet(copySolver<odesolver::RungeKutta4Solver>(solver1), NULL,createConstraintSolver(&solver1, &solver2));
}

typedef linearsolver::CGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector> DefaultCGLinearSolver;

LinearSolver::SPtr createLinearSolver(OdeSolver* solver1, OdeSolver* solver2)
{
    DefaultCGLinearSolver::SPtr lsolver = sofa::core::objectmodel::New<DefaultCGLinearSolver>();
    DefaultCGLinearSolver* lsolver1 = NULL; if (solver1!=NULL) solver1->getContext()->get(lsolver1, core::objectmodel::BaseContext::SearchDown);
    DefaultCGLinearSolver* lsolver2 = NULL; if (solver2!=NULL) solver2->getContext()->get(lsolver2, core::objectmodel::BaseContext::SearchDown);
    unsigned int maxIter = 0;
    double tolerance = 1.0e10;
    double smallDenominatorThreshold = 1.0e10;
    if (lsolver1)
    {
        if (lsolver1->f_maxIter.getValue() > maxIter) maxIter = lsolver1->f_maxIter.getValue();
        if (lsolver1->f_tolerance.getValue() < tolerance) tolerance = lsolver1->f_tolerance.getValue();
        if (lsolver1->f_smallDenominatorThreshold.getValue() > smallDenominatorThreshold) smallDenominatorThreshold = lsolver1->f_smallDenominatorThreshold.getValue();
    }
    if (lsolver2)
    {
        if (lsolver2->f_maxIter.getValue() > maxIter) maxIter = lsolver2->f_maxIter.getValue();
        if (lsolver2->f_tolerance.getValue() < tolerance) tolerance = lsolver2->f_tolerance.getValue();
        if (lsolver2->f_smallDenominatorThreshold.getValue() > smallDenominatorThreshold) smallDenominatorThreshold = lsolver2->f_smallDenominatorThreshold.getValue();
    }
    if (maxIter > 0) lsolver->f_maxIter.setValue( maxIter );
    if (tolerance < 1.0e10) lsolver->f_tolerance.setValue( tolerance );
    if (smallDenominatorThreshold < 1.0e10) lsolver->f_smallDenominatorThreshold.setValue( smallDenominatorThreshold );
    return lsolver;
}

SolverSet createSolverEulerImplicitEulerImplicit(odesolver::EulerImplicitSolver& solver1, odesolver::EulerImplicitSolver& solver2)
{
    odesolver::EulerImplicitSolver::SPtr solver = sofa::core::objectmodel::New<odesolver::EulerImplicitSolver>();
    solver->f_rayleighStiffness.setValue( solver1.f_rayleighStiffness.getValue() < solver2.f_rayleighStiffness.getValue() ? solver1.f_rayleighStiffness.getValue() : solver2.f_rayleighStiffness.getValue() );

    solver->f_rayleighMass.setValue( solver1.f_rayleighMass.getValue() < solver2.f_rayleighMass.getValue() ? solver1.f_rayleighMass.getValue() : solver2.f_rayleighMass.getValue() );
    solver->f_velocityDamping.setValue( solver1.f_velocityDamping.getValue() > solver2.f_velocityDamping.getValue() ? solver1.f_velocityDamping.getValue() : solver2.f_velocityDamping.getValue());
    return SolverSet(solver,
            createLinearSolver(&solver1, &solver2),
            createConstraintSolver(&solver1, &solver2));
}

SolverSet createSolverStaticSolver(odesolver::StaticSolver& solver1, odesolver::StaticSolver& solver2)
{
    return SolverSet(copySolver<odesolver::StaticSolver>(solver1),
            createLinearSolver(&solver1, &solver2),
            createConstraintSolver(&solver1, &solver2));
}

// Then the other, with the policy of taking the more precise solver

SolverSet createSolverRungeKutta4Euler(odesolver::RungeKutta4Solver& solver1, odesolver::EulerSolver& solver2)
{
    return SolverSet(copySolver<odesolver::RungeKutta4Solver>(solver1), NULL,createConstraintSolver(&solver1, &solver2));
}

SolverSet createSolverEulerImplicitEuler(odesolver::EulerImplicitSolver& solver1, odesolver::EulerSolver& solver2)
{
    return SolverSet(copySolver<odesolver::EulerImplicitSolver>(solver1),
            createLinearSolver(&solver1, NULL),
            createConstraintSolver(&solver1, &solver2));
}

SolverSet createSolverEulerImplicitRungeKutta4(odesolver::EulerImplicitSolver& solver1, odesolver::RungeKutta4Solver& solver2)
{
    return SolverSet(copySolver<odesolver::EulerImplicitSolver>(solver1),
            createLinearSolver(&solver1, NULL),
            createConstraintSolver(&solver1, &solver2));
}

}// namespace SolverMergers

using namespace solvermergers;

SolverSet SolverMerger::merge(core::behavior::OdeSolver* solver1, core::behavior::OdeSolver* solver2)
{
    static SolverMerger instance;
    SolverSet obj=instance.solverDispatcher.go(*solver1, *solver2);


    return obj;
}

SolverMerger::SolverMerger()
{
    solverDispatcher.add<odesolver::EulerSolver,odesolver::EulerSolver,createSolverEulerEuler,false>();
    solverDispatcher.add<odesolver::RungeKutta4Solver,odesolver::RungeKutta4Solver,createSolverRungeKutta4RungeKutta4,false>();
    solverDispatcher.add<odesolver::EulerImplicitSolver,odesolver::EulerImplicitSolver,createSolverEulerImplicitEulerImplicit,false>();
    solverDispatcher.add<odesolver::RungeKutta4Solver,odesolver::EulerSolver,createSolverRungeKutta4Euler,true>();
    solverDispatcher.add<odesolver::EulerImplicitSolver,odesolver::EulerSolver,createSolverEulerImplicitEuler,true>();
    solverDispatcher.add<odesolver::EulerImplicitSolver,odesolver::RungeKutta4Solver,createSolverEulerImplicitRungeKutta4,true>();
    solverDispatcher.add<odesolver::StaticSolver,odesolver::StaticSolver,createSolverStaticSolver,true>();
}

}// namespace collision
} // namespace component
} // namespace Sofa
