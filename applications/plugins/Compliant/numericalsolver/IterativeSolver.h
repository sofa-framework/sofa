#ifndef COMPLIANT_ITERATIVESOLVER_H
#define COMPLIANT_ITERATIVESOLVER_H

#include "KKTSolver.h"
#include "Response.h"

namespace sofa {
namespace component {
namespace linearsolver {


// base class for iterative solvers
class SOFA_Compliant_API IterativeSolver : public KKTSolver {
  public:

	IterativeSolver();

	// iterations control
	Data<SReal> precision;
	Data<unsigned> iterations;
	Data<bool> relative;

	// convergence logging
	Data<bool> cv_record;
	Data< vector<SReal> > cv_data;
	
  protected:
	// convenience 
	void cv_clear();
	void cv_push(SReal value);
};

}
}
}

#endif
