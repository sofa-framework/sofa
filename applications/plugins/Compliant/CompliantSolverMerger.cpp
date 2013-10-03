
#include "CompliantSolverMerger.h"

#include <sofa/component/collision/SolverMerger.h>
#include <sofa/helper/FnDispatcher.inl>

#include "AssembledSolver.h"

#include "MinresSolver.h"
#include "CgSolver.h"

#include "LDLTSolver.h"

namespace sofa
{
namespace component
{
namespace collision
{


    core::behavior::BaseLinearSolver::SPtr createCgSolver(linearsolver::CgSolver& solver1, linearsolver::CgSolver& solver2)
    {
        linearsolver::CgSolver::SPtr lsolver = sofa::core::objectmodel::New<linearsolver::CgSolver>();
        lsolver->precision.setValue( std::min(solver1.precision.getValue(),solver2.precision.getValue())  );
        lsolver->relative.setValue( solver1.relative.getValue() || solver2.relative.getValue() );
        lsolver->iterations.setValue( std::max(solver1.iterations.getValue(),solver2.iterations.getValue())  );
        return lsolver;
    }

    core::behavior::BaseLinearSolver::SPtr createMinresSolver(linearsolver::MinresSolver& solver1, linearsolver::MinresSolver& solver2)
    {
        linearsolver::MinresSolver::SPtr lsolver = sofa::core::objectmodel::New<linearsolver::MinresSolver>();
        lsolver->precision.setValue( std::min(solver1.precision.getValue(),solver2.precision.getValue())  );
        lsolver->relative.setValue( solver1.relative.getValue() || solver2.relative.getValue() );
        lsolver->iterations.setValue( std::max(solver1.iterations.getValue(),solver2.iterations.getValue())  );
        return lsolver;
    }

    core::behavior::BaseLinearSolver::SPtr createLDLTSolver(linearsolver::LDLTSolver& solver1, linearsolver::LDLTSolver& solver2)
    {
        linearsolver::LDLTSolver::SPtr lsolver = sofa::core::objectmodel::New<linearsolver::LDLTSolver>();
        lsolver->damping.setValue( std::max(solver1.damping.getValue(),solver2.damping.getValue())  );
        return lsolver;
    }




/////////////////////





    SolverSet createAssembledSolver(odesolver::AssembledSolver& solver1, odesolver::AssembledSolver& solver2)
    {
        odesolver::AssembledSolver::SPtr solver = sofa::core::objectmodel::New<odesolver::AssembledSolver>();

        solver->use_velocity.setValue( solver1.use_velocity.getValue() || solver2.use_velocity.getValue() );
        solver->warm_start.setValue( solver1.warm_start.getValue() && solver2.warm_start.getValue() );
        solver->propagate_lambdas.setValue( solver1.propagate_lambdas.getValue() && solver2.propagate_lambdas.getValue() );
        solver->stabilization.setValue( solver1.stabilization.getValue() || solver2.stabilization.getValue() );

        return SolverSet(solver, CompliantSolverMerger::mergeLinearSolver(&solver1,&solver2) );
    }




////////////////////////






    CompliantSolverMerger::CompliantSolverMerger()
    {
        _linearSolverDispatcher.add<linearsolver::CgSolver,linearsolver::CgSolver,createCgSolver,true>();
        _linearSolverDispatcher.add<linearsolver::MinresSolver,linearsolver::MinresSolver,createMinresSolver,true>();
        _linearSolverDispatcher.add<linearsolver::LDLTSolver,linearsolver::LDLTSolver,createLDLTSolver,true>();
    }

    CompliantSolverMerger* CompliantSolverMerger::getInstance()
    {
        static CompliantSolverMerger instance;
        return &instance;
    }

    core::behavior::BaseLinearSolver::SPtr CompliantSolverMerger::mergeLinearSolver(core::behavior::OdeSolver* solver1, core::behavior::OdeSolver* solver2)
    {
        core::behavior::BaseLinearSolver* lsolver1 = NULL; if (solver1!=NULL) solver1->getContext()->get(lsolver1, core::objectmodel::BaseContext::SearchDown);
        core::behavior::BaseLinearSolver* lsolver2 = NULL; if (solver2!=NULL) solver2->getContext()->get(lsolver2, core::objectmodel::BaseContext::SearchDown);

        if( lsolver1 && lsolver2 )
            return getInstance()->_linearSolverDispatcher.go(*lsolver1, *lsolver2);
        else
            return sofa::core::objectmodel::New<linearsolver::MinresSolver>(); // by default a minressolver with default options
    }

    void CompliantSolverMerger::add()
    {
        SolverMerger::addDispatcher<odesolver::AssembledSolver,odesolver::AssembledSolver,createAssembledSolver,true>();
    }



} // namespace collision
} // namespace component
} // namespace sofa


