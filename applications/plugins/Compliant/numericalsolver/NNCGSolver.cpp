#include "NNCGSolver.h"

#include <sofa/core/ObjectFactory.h>

#include "../utils/scoped.h"

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(NNCGSolver)
int NNCGSolverClass = core::RegisterObject("Nonsmooth Nonlinear Conjugate Gradient Solver").add< NNCGSolver >();


NNCGSolver::NNCGSolver()
    : verbose(initData(&verbose, false, "verbose", "print stuff"))
{}

// TODO copypasta 
void NNCGSolver::solve_impl(vec& res,
                            const system_type& sys,
                            const vec& rhs,
                            bool correct,
                            real /*damping*/ ) const {

	scoped::timer timer("system solve");


	// free velocity
	vec tmp( sys.m );
	
	assert( response );
	response->solve(tmp, sys.P.selfadjointView<Eigen::Upper>() * rhs.head( sys.m ) );
	res.head(sys.m).noalias() = sys.P.selfadjointView<Eigen::Upper>() * tmp;


	if( this->bench ) {
		this->bench->clear();
		this->bench->restart();
    }

	// we're done lol
	if( !sys.n ) return;

	// net constraint correction
	vec net = vec::Zero( sys.m );

	// lagrange multipliers TODO reuse res.tail( sys.n ) ?
	vec lambda = res.tail(sys.n); 
	
	// lambda change
	vec delta = vec::Zero( sys.n );
	
	// constant term in rhs
	vec constant = rhs.tail(sys.n) - sys.J * res.head( sys.m );
	
	// rhs term
	vec error = vec::Zero( sys.n );
	
    const real epsilon = relative.getValue() ? constant.norm() * precision.getValue() : precision.getValue();
    const real epsilon2 = epsilon * epsilon;
	
	// outer loop
	unsigned k = 0;
    vec lambda_prev, grad_prev, p, grad;
    real grad_prev_sqnorm = 0;

	for(unsigned max = iterations.getValue(); k < max; ++k) {
		lambda_prev = lambda;
		
		// compute fresh net correction
		net = mapping_response * lambda;
		
		// step lambda
        /*real estimate = */
		step( lambda, net, sys, constant, error, delta, correct );


		// conjugation
        if( k>0 ) {

            grad = -lambda + lambda_prev;

            assert( grad_prev.norm() > std::numeric_limits<real>::epsilon() );
            real beta = grad.squaredNorm() / grad_prev_sqnorm;

            if( beta > 1 ) {
                if(verbose.getValue()) {
                    serr << "NLCG restart at iteration " << k << sendl;
                }
                // restart
                p.setZero();
            } else {
                // conjugation
                lambda += beta * p;
                p = beta * p - grad;
            }

            grad_prev = grad;
            grad_prev_sqnorm = grad.squaredNorm();
        } else {
            // first iteration
            // (here grad_prev is grad, avoiding a copy grad_dev = grad)
            grad_prev = -lambda + lambda_prev;
            p = -grad_prev;
            grad_prev_sqnorm = grad_prev.squaredNorm();
        }



		if( this->bench ) this->bench->lcp(sys, constant, *response, lambda);



//         if( cb && cb( lambda ) < epsilon ) break;
		// // TODO stop criterion delta.norm() < eps
         if( grad_prev_sqnorm <= epsilon2 ) break; // at that point grad_prev==grad so grad_prev_sqnorm==grad.squaredNorm()
		// if( estimate <= epsilon2 ) break;
	}

	res.head( sys.m ) += net;
	res.tail( sys.n ) = lambda;

}




}
}
}
