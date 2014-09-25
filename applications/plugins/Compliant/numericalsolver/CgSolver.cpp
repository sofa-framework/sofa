#include "CgSolver.h"

#include <sofa/core/ObjectFactory.h>


#include "utils/schur.h"
#include "utils/kkt.h"
#include "utils/cg.h"
#include "utils/preconditionedcg.h"


namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(CgSolver);
int CgSolverClass = core::RegisterObject("Sparse CG linear solver").add< CgSolver >();

CgSolver::CgSolver() 
{
	
}

// TODO: copy pasta; put this in utils (see MinresSolver.cpp)
template<class Params>
static void report(const Params& p) {
	std::cerr << "cg: " << p.iterations 
			  << " iterations, absolute residual: " << p.precision << std::endl;
}

// delicious copypasta (see minres) TODO factor this in utils
void CgSolver::solve_schur(AssembledSystem::vec& x,
						   const AssembledSystem& sys,
						   const AssembledSystem::vec& b,
						   real damping) const {

	// unconstrained velocity
	vec tmp(sys.m);
	response->solve(tmp, b.head(sys.m));
	x.head( sys.m ) = tmp;
	
	if( sys.n ) {
		
		::schur<response_type> A(sys, *response, damping);
		
		vec rhs = b.tail(sys.n) - sys.J * x.head(sys.m);
		
		vec lambda = x.tail(sys.n);

		typedef ::cg<real> solver_type;		
		
		solver_type::params p = params(rhs);
		solver_type::solve(lambda, A, rhs, p);
		
		// constraint velocity correction
		response->solve(tmp, sys.J.transpose() * lambda );

		x.head( sys.m ) += tmp;
		x.tail( sys.n ) = lambda;
		
	}

}


void CgSolver::solve_kkt(AssembledSystem::vec& x,
                         const AssembledSystem& system,
                         const AssembledSystem::vec& b,
						 real /* damping */ ) const {
	if( system.n ) {
		throw std::logic_error("CG can't solve KKT system with constraints. you need to turn on schur and add a response component for this");
	}

    params_type p = params(b);

    kkt::matrixQ A(system);

    typedef ::cg<real> solver_type;
    solver_type::solve(x, A, b, p);

    report("cg (kkt)", p );

}



}
}
}


