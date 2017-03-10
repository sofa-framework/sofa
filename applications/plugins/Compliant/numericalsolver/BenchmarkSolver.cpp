#include "BenchmarkSolver.h"

#include <sofa/core/ObjectFactory.h>


namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(BenchSolver)
int BenchmarkSolverClass = core::RegisterObject("Benchmark solver: runs other KKTSolvers successively on a given problem").add< BenchmarkSolver >();

BenchmarkSolver::BenchmarkSolver() { }

void BenchmarkSolver::init() {

	solvers.clear();
	this->getContext()->get<KKTSolver>( &solvers, core::objectmodel::BaseContext::Local );
	
	if( solvers.size() < 2 ) {
		std::cerr << "warning: no other kkt solvers found" << std::endl;
	} else {
		std::cout << "BenchmarkSolver: dynamics/correction will use " 
				  << solvers.back()->getName() 
				  << std::endl;
	}
}

void BenchmarkSolver::factor(const system_type& system) {

	// start at 1 since 0 is current solver
    for( unsigned i = 1, n = solvers.size() ; i < n; ++i ) {
		solvers[i]->factor(system);
	}
	
}


// solution is that of the first solver
void BenchmarkSolver::correct(vec& res,
                              const system_type& sys,
                              const vec& rhs, real damping) const {
	assert( solvers.size() > 1 );
    solvers.back()->correct(res, sys, rhs, damping);
}

// solution is that of the last solver
void BenchmarkSolver::solve(vec& res,
							const system_type& sys,
							const vec& rhs) const {

    vec backup = res; // backup initial solution

	// start at 1 since 0 is current solver
    for( unsigned i = 1, n = solvers.size(); i < n; ++i ) {
		res = backup;
		solvers[i]->solve(res, sys, rhs);
	}
	
}






}
}
}
