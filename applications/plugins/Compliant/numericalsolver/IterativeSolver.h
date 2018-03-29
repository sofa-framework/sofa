#ifndef COMPLIANT_ITERATIVESOLVER_H
#define COMPLIANT_ITERATIVESOLVER_H

#include "KKTSolver.h"
#include "Benchmark.h"

namespace sofa {
namespace component {
namespace linearsolver {


/// Base class for iterative solvers
class SOFA_Compliant_API IterativeSolver : public KKTSolver {
  public:

	IterativeSolver();

	// iterations control
	Data<SReal> precision;
	Data<unsigned> iterations;
	Data<bool> relative; ///< use relative precision

  protected:
    SingleLink<IterativeSolver, Benchmark, 0> bench;
};

}
}
}

#endif
