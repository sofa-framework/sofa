#include "CgSolver.h"

#include <sofa/core/ObjectFactory.h>


#include "utils/schur.h"
#include "utils/kkt.h"
#include "utils/cg.h"
#include "utils/preconditionedcg.h"


namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(CgSolver)
int CgSolverClass = core::RegisterObject("Sparse CG linear solver").add< CgSolver >();

CgSolver::CgSolver() 
{
	
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

        report("cg (schur)", p );
	}

}

// this code could also be factorized with minres code (maybe in KrylovSolver?)
void CgSolver::solve_kkt(AssembledSystem::vec& x,
                         const AssembledSystem& system,
                         const AssembledSystem::vec& b,
                         real damping ) const {

    params_type p = params(b);

    vec rhs = b;
    if( system.n ) rhs.tail(system.n) = -rhs.tail(system.n);

    kkt A(system, false, damping);

    typedef ::cg<real> solver_type;
    solver_type::solve(x, A, rhs, p);

    report("cg (kkt)", p );
}



}
}
}


