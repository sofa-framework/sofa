#include "IterativeSolver.h"
#include "../utils/edit.h"


namespace sofa {
namespace component {
namespace linearsolver {

IterativeSolver::IterativeSolver() 
	: precision(initData(&precision, 
	                     SReal(1e-3),
	                     "precision",
	                     "residual norm threshold")),
	  iterations(initData(&iterations,
	                      unsigned(10),
	                      "iterations",
	                      "iteration bound")),
	  relative(initData(&relative, false, "relative", "use relative precision") ),

	  cv_record(initData(&cv_record, false, "cv_record", "record convergence during solve")),
	  cv_data(initData(&cv_data, "cv_data", "recorded convergence values, if any (read-only)")) {


}


void IterativeSolver::cv_clear() {
	if( cv_record.getValue() ) {
		edit(cv_data)->clear();
		edit(cv_data)->reserve( iterations.getValue() );
	}
}


void IterativeSolver::cv_push(SReal value) {
	if( cv_record.getValue() ) {
		edit(cv_data)->push_back( value );
	}
}


}
}
}
