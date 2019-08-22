#include "IterativeSolver.h"


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

      bench(BaseLink::InitLink<IterativeSolver>(this, "bench", "benchmark component to record convergence"))
{}



}
}
}
