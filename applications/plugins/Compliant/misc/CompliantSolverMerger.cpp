
#include "CompliantSolverMerger.h"

#include <SofaMiscCollision/SolverMerger.h>
#include <sofa/helper/FnDispatcher.inl>

#include "odesolver/CompliantImplicitSolver.h"
#include "odesolver/CompliantNLImplicitSolver.h"

#include "numericalsolver/MinresSolver.h"
#include "numericalsolver/CgSolver.h"
#include "numericalsolver/LDLTSolver.h"
#include "numericalsolver/SequentialSolver.h"

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

    core::behavior::BaseLinearSolver::SPtr createLDLTSolver(linearsolver::LDLTSolver& /*solver1*/, linearsolver::LDLTSolver& /*solver2*/)
    {
        linearsolver::LDLTSolver::SPtr lsolver = sofa::core::objectmodel::New<linearsolver::LDLTSolver>();
//        lsolver->damping.setValue( std::max(solver1.damping.getValue(),solver2.damping.getValue())  );
        return lsolver;
    }

    core::behavior::BaseLinearSolver::SPtr createSequentialSolver(linearsolver::SequentialSolver& solver1, linearsolver::SequentialSolver& solver2)
    {
        linearsolver::SequentialSolver::SPtr lsolver = sofa::core::objectmodel::New<linearsolver::SequentialSolver>();
        lsolver->precision.setValue( std::min(solver1.precision.getValue(),solver2.precision.getValue())  );
        lsolver->relative.setValue( solver1.relative.getValue() || solver2.relative.getValue() );
        lsolver->iterations.setValue( std::max(solver1.iterations.getValue(),solver2.iterations.getValue())  );
        lsolver->omega.setValue( std::min(solver1.omega.getValue(),solver2.omega.getValue())  );
        return lsolver;
    }




/////////////////////


    // TODO adjust which parameters must be merged


    SolverSet createCompliantImplicitSolver(odesolver::CompliantImplicitSolver& solver1, odesolver::CompliantImplicitSolver& solver2)
    {
        odesolver::CompliantImplicitSolver::SPtr solver = sofa::core::objectmodel::New<odesolver::CompliantImplicitSolver>();

        solver->warm_start.setValue( solver1.warm_start.getValue() && solver2.warm_start.getValue() );
        solver->propagate_lambdas.setValue( solver1.propagate_lambdas.getValue() && solver2.propagate_lambdas.getValue() );
        solver->stabilization.beginEdit()->setSelectedItem( std::max( solver1.stabilization.getValue().getSelectedId(), solver2.stabilization.getValue().getSelectedId() ) ); solver->stabilization.endEdit();

        return SolverSet(solver, CompliantSolverMerger::mergeLinearSolver(&solver1,&solver2) );
    }

    SolverSet createCompliantNLImplicitSolver(odesolver::CompliantNLImplicitSolver& solver1, odesolver::CompliantNLImplicitSolver& solver2)
    {
        odesolver::CompliantNLImplicitSolver::SPtr solver = sofa::core::objectmodel::New<odesolver::CompliantNLImplicitSolver>();

        solver->warm_start.setValue( solver1.warm_start.getValue() && solver2.warm_start.getValue() );
        solver->propagate_lambdas.setValue( solver1.propagate_lambdas.getValue() && solver2.propagate_lambdas.getValue() );
        solver->stabilization.beginEdit()->setSelectedItem( std::max( solver1.stabilization.getValue().getSelectedId(), solver2.stabilization.getValue().getSelectedId() ) ); solver->stabilization.endEdit();

        return SolverSet(solver, CompliantSolverMerger::mergeLinearSolver(&solver1,&solver2) );
    }

    SolverSet createCompliantNLImplicitSolver(odesolver::CompliantImplicitSolver& solver1, odesolver::CompliantNLImplicitSolver& solver2)
    {
        odesolver::CompliantNLImplicitSolver::SPtr solver = sofa::core::objectmodel::New<odesolver::CompliantNLImplicitSolver>();

        solver->warm_start.setValue( solver1.warm_start.getValue() && solver2.warm_start.getValue() );
        solver->propagate_lambdas.setValue( solver1.propagate_lambdas.getValue() && solver2.propagate_lambdas.getValue() );
        solver->stabilization.beginEdit()->setSelectedItem( std::max( solver1.stabilization.getValue().getSelectedId(), solver2.stabilization.getValue().getSelectedId() ) ); solver->stabilization.endEdit();

        return SolverSet(solver, CompliantSolverMerger::mergeLinearSolver(&solver1,&solver2) );
    }

////////////////////////






    CompliantSolverMerger::CompliantSolverMerger()
    {
        _linearSolverDispatcher.add<linearsolver::CgSolver,linearsolver::CgSolver,createCgSolver,true>();
        _linearSolverDispatcher.add<linearsolver::MinresSolver,linearsolver::MinresSolver,createMinresSolver,true>();
        _linearSolverDispatcher.add<linearsolver::LDLTSolver,linearsolver::LDLTSolver,createLDLTSolver,true>();
        _linearSolverDispatcher.add<linearsolver::SequentialSolver,linearsolver::SequentialSolver,createSequentialSolver,true>();
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
        SolverMerger::addDispatcher<odesolver::CompliantImplicitSolver,odesolver::CompliantImplicitSolver,createCompliantImplicitSolver,true>();
        SolverMerger::addDispatcher<odesolver::CompliantNLImplicitSolver,odesolver::CompliantNLImplicitSolver,createCompliantNLImplicitSolver,true>();
        SolverMerger::addDispatcher<odesolver::CompliantImplicitSolver,odesolver::CompliantNLImplicitSolver,createCompliantNLImplicitSolver,false>();
    }



} // namespace collision
} // namespace component
} // namespace sofa


