#include "MinresSolver.h"

#include <sofa/core/ObjectFactory.h>

#include "utils/scoped.h"
#include "utils/minres.h"

#include "utils/kkt.h"
#include "utils/schur.h"

namespace sofa {
namespace component {
namespace linearsolver {
 
SOFA_DECL_CLASS(MinresSolver);
int MinresSolverClass = core::RegisterObject("Sparse Minres linear solver").add< MinresSolver >();


MinresSolver::MinresSolver() 
	: // use_schur(initData(&use_schur, false, "use_schur", "use Schur complement when solving. *warning* this will invert the response matrix at each time step unless fast_schur is true")),
	  // fast_schur(initData(&fast_schur, false, "fast_schur", "only invert response matrix once")),
	parallel(initData(&parallel, false, "parallel", "use openmp to parallelize matrix-vector products when use_schur is false"))
{
	
}

			
			
void MinresSolver::factor(const AssembledSystem& sys) {

	// if( use_schur.getValue() ) {

	// 	// TODO is there a conversion between rmat and cmat ?
	// 	response.compute(sys.H);
		
	// }

}

template<class Params>
static void report(const Params& p) {
	std::cerr << "minres: " << p.iterations << " iterations, absolute residual: " << p.precision << std::endl;
}

// void MinresSolver::solve_schur(AssembledSystem::vec& x,
//                                const AssembledSystem& sys,
//                                const AssembledSystem::vec& b) const {
// 	// unconstrained velocity
// 	x.head( sys.m ) = response.solve(b.head(sys.m));
	
// 	if( sys.n ) {

// 		schur<response_type> A(sys, response);
		
// 		vec rhs = b.tail(sys.n) - sys.J * x.head(sys.m);
		
// 		vec lambda = x.tail(sys.n);

// 		typedef ::minres<SReal> solver_type;		
// 		solver_type::params p;
// 		p.iterations = iterations.getValue();
// 		p.precision = precision.getValue();
		
// 		solver_type::solve(lambda, A, rhs, p);
		
// 		x.head( sys.m ) += response.solve( sys.J.transpose() * lambda );
// 		x.tail( sys.n ) = lambda;
		
// 		if( verbose.getValue() ) report( p );
// 	}


// }


void MinresSolver::solve_kkt(AssembledSystem::vec& x,
                             const AssembledSystem& system,
                             const AssembledSystem::vec& b) const {
	params_type p = params(b);
			
	vec rhs = b;
	if( system.n ) rhs.tail(system.n) = -rhs.tail(system.n);
	
	kkt A(system, parallel.getValue() );
	
	typedef minres<real> solver_type;
	solver_type::solve(x, A, rhs, p);
	
	if( verbose.getValue() ) report( p );
}


void MinresSolver::solve(AssembledSystem::vec& x,
                             const AssembledSystem& system,
                             const AssembledSystem::vec& b) const {
	solve_kkt(x, system, b);

	// if(use_schur.getValue() ) {
	// 	solve_schur(x, system, b);
	// } else {
	// 	solve_kkt(x, system, b);
	// }
	
}
			
}
}
}


