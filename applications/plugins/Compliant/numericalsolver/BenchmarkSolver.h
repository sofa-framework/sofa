#ifndef COMPLIANT_BENCHMARKSOLVER_H
#define COMPLIANT_BENCHMARKSOLVER_H


#include <Compliant/Compliant.h>

#include "KKTSolver.h"

namespace sofa {
namespace component {
namespace linearsolver {

// simple solver that discovers kktsolvers at its level, then launch
// them sequentially on each kkt problem
class SOFA_Compliant_API BenchmarkSolver : public KKTSolver {
  public:

	SOFA_CLASS(BenchmarkSolver, KKTSolver);
	 
	BenchmarkSolver();
	
	virtual void init();

	virtual void factor(const system_type& system);
	
	virtual void solve(vec& x,
	                   const system_type& system,
	                   const vec& rhs) const;

	virtual void correct(vec& x,
						 const system_type& system,
						 const vec& rhs) const;

  protected:
    std::vector< KKTSolver::SPtr > solvers;

};


}
}
}

#endif
