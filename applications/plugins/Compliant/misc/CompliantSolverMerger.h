#ifndef __COMPLIANT_COMPLIANTSOLVERMERGER_H__
#define __COMPLIANT_COMPLIANTSOLVERMERGER_H__

#include <Compliant/config.h>
#include <sofa/core/behavior/LinearSolver.h>

namespace sofa
{

namespace core
{
namespace behavior
{
    class OdeSolver;
} // namespace behavior
} // namespace core

namespace component
{
namespace collision
{


    class SOFA_Compliant_API CompliantSolverMerger
    {
    public:

        static void add();

        static core::behavior::BaseLinearSolver::SPtr mergeLinearSolver(core::behavior::OdeSolver* solver1, core::behavior::OdeSolver* solver2);

        template<typename SolverType1, typename SolverType2, core::behavior::BaseLinearSolver::SPtr (*F)(SolverType1&,SolverType2&),bool symmetric> static void addLinearSolverDispatcher()
        {
            getInstance()->_linearSolverDispatcher.add<SolverType1,SolverType2,F,symmetric>();
        }

    protected:

        static CompliantSolverMerger* getInstance();

        helper::FnDispatcher<core::behavior::BaseLinearSolver, core::behavior::BaseLinearSolver::SPtr> _linearSolverDispatcher;

        CompliantSolverMerger();
    };





} // namespace collision
} // namespace component
} // namespace sofa


#endif // __COMPLIANT_COMPLIANTSOLVERMERGER_H__
