#include "KrylovSolver.h"


namespace sofa {
namespace component {
namespace linearsolver {


KrylovSolver::KrylovSolver() 
	: precision(initData(&precision, 
	                     SReal(1e-3),
	                     "precision",
	                     "residual norm threshold")),
	  iterations(initData(&iterations,
	                      unsigned(10),
	                      "iterations",
	                      "iteration bound")),
	  relative(initData(&relative, false, "relative", "use relative precision") ),
	  schur(initData(&schur, false, "schur", "perform solving on the schur complement. you *must* have a response component nearby in the graph.")),
	  verbose(initData(&verbose, false, "verbose", "print debug stuff on std::cerr") )
{
	
}

void KrylovSolver::init() {
	
    KKTSolver::init();

	if( schur.getValue() ) {
		response = this->getContext()->get<Response>(core::objectmodel::BaseContext::Local);
		
		if(!response) throw std::logic_error("response component not found, you need one next to the KKTSolver");
		
	}

}


void KrylovSolver::factor(const system_type& sys) {
	if( response ) response->factor( sys.H );
}


KrylovSolver::params_type KrylovSolver::params(const vec& rhs) const {

	params_type res;
	res.iterations = iterations.getValue();
	res.precision = precision.getValue();
				
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

void KrylovSolver::solveWithPreconditioner(vec& x,
                         const system_type& system,
                         const vec& rhs) const {
    if( schur.getValue() ) {
        assert( response );
        solve_schur(x, system, rhs);
    } else {
        if( _preconditioner )
            solve_kkt_with_preconditioner(x, system, rhs);
        else
            solve_kkt(x, system, rhs);
    }
}


void KrylovSolver::report(const char* what, const params_type& p) const {
	if( verbose.getValue() ) {
		std::cerr << what << "- iterations: " << p.iterations << ", (abs) residual: " << p.precision << std::endl;
	}
}



}
}
}


