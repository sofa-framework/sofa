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

#include <sofa/helper/FnDispatcher.h>
#include <sofa/helper/FnDispatcher.inl>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/component/odesolver/RungeKutta4Solver.h>
#include <sofa/component/odesolver/CGImplicitSolver.h>
#include <sofa/component/odesolver/StaticSolver.h>
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>

namespace sofa
{

namespace component
{

namespace collision
{
using sofa::core::componentmodel::behavior::OdeSolver;
using sofa::core::componentmodel::behavior::LinearSolver;

namespace solvermergers
{

template<class T>
T* copySolver(const T& s)
{
    const T* src = &s;
    T* res = new T;
    for (unsigned int i=0; i<src->getFields().size(); ++i)
    {
        core::objectmodel::BaseData* s = src->getFields()[i].second;
        core::objectmodel::BaseData* d = res->findField(src->getFields()[i].first);
        if (d)
            d->copyValue(s);
    }
    return res;
}

// First the easy cases...

SolverSet createSolverEulerEuler(odesolver::EulerSolver& solver1, odesolver::EulerSolver& /*solver2*/)
{
    return SolverSet(copySolver<odesolver::EulerSolver>(solver1), NULL);
}

SolverSet createSolverRungeKutta4RungeKutta4(odesolver::RungeKutta4Solver& solver1, odesolver::RungeKutta4Solver& /*solver2*/)
{
    return SolverSet(copySolver<odesolver::RungeKutta4Solver>(solver1), NULL);
}

SolverSet createSolverCGImplicitCGImplicit(odesolver::CGImplicitSolver& solver1, odesolver::CGImplicitSolver& solver2)
{
    odesolver::CGImplicitSolver* solver = new odesolver::CGImplicitSolver;
    solver->f_maxIter.setValue( solver1.f_maxIter.getValue() > solver2.f_maxIter.getValue() ? solver1.f_maxIter.getValue() : solver2.f_maxIter.getValue() );
    solver->f_tolerance.setValue( solver1.f_tolerance.getValue() < solver2.f_tolerance.getValue() ? solver1.f_tolerance.getValue() : solver2.f_tolerance.getValue());
    solver->f_smallDenominatorThreshold.setValue( solver1.f_smallDenominatorThreshold.getValue() < solver2.f_smallDenominatorThreshold.getValue() ? solver1.f_smallDenominatorThreshold.getValue() : solver2.f_smallDenominatorThreshold.getValue());

    solver->f_rayleighStiffness.setValue( solver1.f_rayleighStiffness.getValue() < solver2.f_rayleighStiffness.getValue() ? solver1.f_rayleighStiffness.getValue() : solver2.f_rayleighStiffness.getValue() );

    solver->f_rayleighMass.setValue( solver1.f_rayleighMass.getValue() < solver2.f_rayleighMass.getValue() ? solver1.f_rayleighMass.getValue() : solver2.f_rayleighMass.getValue() );
    solver->f_velocityDamping.setValue( solver1.f_velocityDamping.getValue() > solver2.f_velocityDamping.getValue() ? solver1.f_velocityDamping.getValue() : solver2.f_velocityDamping.getValue());
    return SolverSet(solver, NULL);
}

typedef linearsolver::CGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector> DefaultCGLinearSolver;

LinearSolver* createLinearSolver(OdeSolver* solver1, OdeSolver* solver2)
{
    DefaultCGLinearSolver* lsolver = new DefaultCGLinearSolver;
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
    odesolver::EulerImplicitSolver* solver = new odesolver::EulerImplicitSolver;
    solver->f_rayleighStiffness.setValue( solver1.f_rayleighStiffness.getValue() < solver2.f_rayleighStiffness.getValue() ? solver1.f_rayleighStiffness.getValue() : solver2.f_rayleighStiffness.getValue() );

    solver->f_rayleighMass.setValue( solver1.f_rayleighMass.getValue() < solver2.f_rayleighMass.getValue() ? solver1.f_rayleighMass.getValue() : solver2.f_rayleighMass.getValue() );
    solver->f_velocityDamping.setValue( solver1.f_velocityDamping.getValue() > solver2.f_velocityDamping.getValue() ? solver1.f_velocityDamping.getValue() : solver2.f_velocityDamping.getValue());
    return SolverSet(solver, createLinearSolver(&solver1, &solver2));
}

SolverSet createSolverStaticSolver(odesolver::StaticSolver& solver1, odesolver::StaticSolver& solver2)
{
    return SolverSet(copySolver<odesolver::StaticSolver>(solver1), createLinearSolver(&solver1, &solver2));
}

// Then the other, with the policy of taking the more precise solver

SolverSet createSolverRungeKutta4Euler(odesolver::RungeKutta4Solver& solver1, odesolver::EulerSolver& /*solver2*/)
{
    return SolverSet(copySolver<odesolver::RungeKutta4Solver>(solver1), NULL);
}

SolverSet createSolverCGImplicitEuler(odesolver::CGImplicitSolver& solver1, odesolver::EulerSolver& /*solver2*/)
{
    return SolverSet(copySolver<odesolver::CGImplicitSolver>(solver1), NULL);
}

SolverSet createSolverCGImplicitRungeKutta4(odesolver::CGImplicitSolver& solver1, odesolver::RungeKutta4Solver& /*solver2*/)
{
    return SolverSet(copySolver<odesolver::CGImplicitSolver>(solver1), NULL);
}

SolverSet createSolverEulerImplicitEuler(odesolver::EulerImplicitSolver& solver1, odesolver::EulerSolver& /*solver2*/)
{
    return SolverSet(copySolver<odesolver::EulerImplicitSolver>(solver1), createLinearSolver(&solver1, NULL));
}

SolverSet createSolverEulerImplicitRungeKutta4(odesolver::EulerImplicitSolver& solver1, odesolver::RungeKutta4Solver& /*solver2*/)
{
    return SolverSet(copySolver<odesolver::EulerImplicitSolver>(solver1), createLinearSolver(&solver1, NULL));
}

SolverSet createSolverEulerImplicitCGImplicit(odesolver::EulerImplicitSolver& solver1, odesolver::CGImplicitSolver& /*solver2*/)
{
    return SolverSet(copySolver<odesolver::EulerImplicitSolver>(solver1), createLinearSolver(&solver1, NULL));
}

}// namespace SolverMergers

using namespace solvermergers;

SolverSet SolverMerger::merge(core::componentmodel::behavior::OdeSolver* solver1, core::componentmodel::behavior::OdeSolver* solver2)
{
    static SolverMerger instance;
    SolverSet obj=instance.solverDispatcher.go(*solver1, *solver2);
#ifdef SOFA_HAVE_EIGEN2
    obj.first->constraintAcc.setValue( (solver1->constraintAcc.getValue() || solver2->constraintAcc.getValue() ) );
    obj.first->constraintVel.setValue( (solver1->constraintVel.getValue() || solver2->constraintVel.getValue() ) );
    obj.first->constraintPos.setValue( (solver1->constraintPos.getValue() || solver2->constraintPos.getValue() ) );
    obj.first->numIterations.setValue( std::max(solver1->numIterations.getValue(), solver2->numIterations.getValue() ) );
    obj.first->maxError.setValue( std::min(solver1->maxError.getValue(), solver2->maxError.getValue() ) );
#endif
    return obj;
}

SolverMerger::SolverMerger()
{
    solverDispatcher.add<odesolver::EulerSolver,odesolver::EulerSolver,createSolverEulerEuler,false>();
    solverDispatcher.add<odesolver::RungeKutta4Solver,odesolver::RungeKutta4Solver,createSolverRungeKutta4RungeKutta4,false>();
    solverDispatcher.add<odesolver::CGImplicitSolver,odesolver::CGImplicitSolver,createSolverCGImplicitCGImplicit,false>();
    solverDispatcher.add<odesolver::EulerImplicitSolver,odesolver::EulerImplicitSolver,createSolverEulerImplicitEulerImplicit,false>();
    solverDispatcher.add<odesolver::RungeKutta4Solver,odesolver::EulerSolver,createSolverRungeKutta4Euler,true>();
    solverDispatcher.add<odesolver::CGImplicitSolver,odesolver::EulerSolver,createSolverCGImplicitEuler,true>();
    solverDispatcher.add<odesolver::CGImplicitSolver,odesolver::RungeKutta4Solver,createSolverCGImplicitRungeKutta4,true>();
    solverDispatcher.add<odesolver::EulerImplicitSolver,odesolver::EulerSolver,createSolverEulerImplicitEuler,true>();
    solverDispatcher.add<odesolver::EulerImplicitSolver,odesolver::RungeKutta4Solver,createSolverEulerImplicitRungeKutta4,true>();
    solverDispatcher.add<odesolver::EulerImplicitSolver,odesolver::CGImplicitSolver,createSolverEulerImplicitCGImplicit,true>();
    solverDispatcher.add<odesolver::StaticSolver,odesolver::StaticSolver,createSolverStaticSolver,true>();
}

}// namespace collision
} // namespace component
} // namespace Sofa
