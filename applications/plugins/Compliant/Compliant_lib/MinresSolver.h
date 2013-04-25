#ifndef MINRESSOLVER_H
#define MINRESSOLVER_H

#include "KKTSolver.h"

#include "AssembledSystem.h"
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa {
namespace component {
namespace linearsolver {
			
// Solver for an AssembledSystem (could be non-linear in case of
// inequalities). This will eventually serve as a base class for
// all kinds of derived solver (sparse cholesky, minres, qp)
			
// TODO: base + derived classes (minres/cholesky/unilateral)
class SOFA_Compliant_API MinresSolver : public KKTSolver {
  public:
	SOFA_CLASS(MinresSolver, KKTSolver);
	
	MinresSolver();				
	
	typedef AssembledSystem::vec vec;

	// solve the KKT system: \mat{ M - h^2 K & J^T \\ J, -C } x = rhs
	// (watch out for the compliance scaling)
	virtual void solve(vec& x,
	                   const AssembledSystem& system,
	                   const vec& rhs) const;

	virtual void factor(const AssembledSystem& system);
				
  public:
	Data<SReal> precision;
	Data<unsigned> iterations;
	Data<bool> relative;

	Data<bool> verbose;
};


}
}
}

#endif
