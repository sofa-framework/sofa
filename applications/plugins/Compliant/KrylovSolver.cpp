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
	  relative(initData(&relative, true, "relative", "use relative precision") ),
	  verbose(initData(&verbose, false, "verbose", "print debug stuff on std::cerr") )
{
	
}


KrylovSolver::params_type KrylovSolver::params(const vec& rhs) const {

	params_type res;
	res.iterations = iterations.getValue();
	res.precision = precision.getValue();
				
	if( relative.getValue() ) res.precision *= rhs.norm();

	return res;
}


}
}
}


