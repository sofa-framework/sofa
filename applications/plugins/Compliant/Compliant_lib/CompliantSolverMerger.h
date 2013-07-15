#ifndef __COMPLIANT_COMPLIANTSOLVERMERGER_H__
#define __COMPLIANT_COMPLIANTSOLVERMERGER_H__


#include <sofa/component/collision/SolverMerger.h>

#include "AssembledSolver.h"
#include "ComplianceSolver.h"

#include "MinresSolver.h"
#include "LDLTSolver.h"

namespace sofa
{
namespace component
{
namespace collision
{



    SolverSet createComplianceSolver(odesolver::ComplianceSolver& solver1, odesolver::ComplianceSolver& solver2)
    {
        odesolver::ComplianceSolver::SPtr solver = sofa::core::objectmodel::New<odesolver::ComplianceSolver>();
//        solver->f_rayleighStiffness.setValue( solver1.f_rayleighStiffness.getValue() < solver2.f_rayleighStiffness.getValue() ? solver1.f_rayleighStiffness.getValue() : solver2.f_rayleighStiffness.getValue() );

//        solver->f_rayleighMass.setValue( solver1.f_rayleighMass.getValue() < solver2.f_rayleighMass.getValue() ? solver1.f_rayleighMass.getValue() : solver2.f_rayleighMass.getValue() );
//        solver->f_velocityDamping.setValue( solver1.f_velocityDamping.getValue() > solver2.f_velocityDamping.getValue() ? solver1.f_velocityDamping.getValue() : solver2.f_velocityDamping.getValue());
        return SolverSet(solver,
                0,
                0 );
    }


    core::behavior::BaseLinearSolver::SPtr createCompliantLinearSolver(const core::behavior::OdeSolver& solver1, const core::behavior::OdeSolver& solver2)
    {
        linearsolver::MinresSolver::SPtr lsolver = sofa::core::objectmodel::New<linearsolver::MinresSolver>();

//        core::behavior::LinearSolver* lsolver1 = NULL; if (solver1!=NULL) solver1->getContext()->get(lsolver1, core::objectmodel::BaseContext::SearchDown);
//        core::behavior::LinearSolver* lsolver2 = NULL; if (solver2!=NULL) solver2->getContext()->get(lsolver2, core::objectmodel::BaseContext::SearchDown);

        // TODO  handle all cases of input linearsolver + data values, etc....

        return lsolver;
    }


    SolverSet createAssembledSolver(odesolver::AssembledSolver& solver1, odesolver::AssembledSolver& solver2)
    {
        odesolver::AssembledSolver::SPtr solver = sofa::core::objectmodel::New<odesolver::AssembledSolver>();

        solver->use_velocity.setValue( solver1.use_velocity.getValue() || solver2.use_velocity.getValue() );
        solver->warm_start.setValue( solver1.warm_start.getValue() && solver2.warm_start.getValue() );
        solver->propagate_lambdas.setValue( solver1.propagate_lambdas.getValue() && solver2.propagate_lambdas.getValue() );
        solver->stabilization.setValue( solver1.stabilization.getValue() || solver2.stabilization.getValue() );

        return SolverSet(solver,
                createCompliantLinearSolver(solver1,solver2),
                0 );
    }

    void addCompliantSolverMerger()
    {
        SolverMerger::addDispatcher<odesolver::ComplianceSolver,odesolver::ComplianceSolver,createComplianceSolver,true>();
        SolverMerger::addDispatcher<odesolver::AssembledSolver,odesolver::AssembledSolver,createAssembledSolver,true>();
    }



} // namespace component
} // namespace collision
} // namespace sofa


#endif // __COMPLIANT_COMPLIANTSOLVERMERGER_H__
