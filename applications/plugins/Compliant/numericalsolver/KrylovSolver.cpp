#include "KrylovSolver.h"

#include "utils/kkt.h"
#include "utils/schur.h"


namespace sofa {
namespace component {
namespace linearsolver {


KrylovSolver::KrylovSolver() 
    : schur(initData(&schur, false, "schur", "perform solving on the schur complement. you *must* have a response component nearby in the graph."))
    , verbose(initData(&verbose, false, "verbose", "print debug stuff on std::cerr") )
    , restart(initData(&restart, unsigned(0), "restart", "restart every n steps"))
    , parallel(initData(&parallel, false, "parallel", "use openmp to parallelize matrix-vector products when use_schur is false (parallelization per KKT blocks, 4 threads)"))
{}

void KrylovSolver::init() {

    IterativeSolver::init();

	if( schur.getValue() ) {
		response = this->getContext()->get<Response>(core::objectmodel::BaseContext::Local);
		
		if(!response) {
            serr << "response component not found, you need one next to the KKTSolver" << sendl;
        }		
    }
    
}


void KrylovSolver::factor(const system_type& sys) {
	if( response && schur.getValue() ) {
        sub = SubKKT(sys);
        sub.factor(*response);
    }
}


KrylovSolver::params_type KrylovSolver::params(const vec& rhs) const {

	params_type res;
	res.iterations = iterations.getValue();
	res.precision = precision.getValue();
	res.restart = restart.getValue();

	if( relative.getValue() ) res.precision *= rhs.norm();
    
	return res;
}


void KrylovSolver::solve(vec& x,
                         const system_type& system,
                         const vec& rhs) const {
	if( schur.getValue() ) {
		assert( response );
		solve_schur(x, system, rhs);
    } else {
        solve_kkt(x, system, rhs);
	}
}


void KrylovSolver::correct(vec& x,
						   const system_type& system,
						   const vec& rhs,
						   real damping) const {
	if( schur.getValue() ) {
		assert( response );
		solve_schur(x, system, rhs, damping);
    } else {
		solve_kkt(x, system, rhs, damping);
	}
	
}


void KrylovSolver::report(const char* what, const params_type& p) const {
	if( verbose.getValue() ) {
		std::cerr << what << "- iterations: " << p.iterations << ", (abs) residual: " << p.precision << std::endl;
	}
}




void KrylovSolver::solve_schur(AssembledSystem::vec& x,
                               const AssembledSystem& sys,
                               const AssembledSystem::vec& b,
							   real damping) const {
	// unconstrained velocity
	vec tmp(sys.m);

    sub.solve(*response, tmp, b.head(sys.m));
	// response->solve(tmp, b.head(sys.m));
    
	x.head( sys.m ) = tmp;
	
	if( sys.n ) {

        SubKKT::Adaptor adaptor = sub.adapt(*response);
        
		::schur<SubKKT::Adaptor> A(sys, adaptor, damping);
		
		vec rhs = b.tail(sys.n) - sys.J * x.head(sys.m);
		
		vec lambda = x.tail(sys.n);

		params_type p = params(rhs);

        // actual method
        solve_schur_impl(lambda, A, rhs, p);
		
		// constraint velocity correction
        sub.solve(*response, tmp, sys.J.transpose() * lambda);
        // response->solve(tmp, sys.J.transpose() * lambda );
        
		x.head( sys.m ) += tmp;
		x.tail( sys.n ) = lambda;
		
		report("schur", p );
	}


}


void KrylovSolver::solve_kkt(AssembledSystem::vec& x,
                             const AssembledSystem& system,
                             const AssembledSystem::vec& b,
							 real damping) const {
	params_type p = params(b);
			
	vec rhs = b;
	if( system.n ) rhs.tail(system.n) = -rhs.tail(system.n);
	
	kkt A(system, parallel.getValue(), damping);
	
    solve_kkt_impl(x, A, rhs, p);
    
	report("kkt", p );
}




}
}
}


