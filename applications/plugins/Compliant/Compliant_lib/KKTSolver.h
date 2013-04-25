#ifndef KKTSOLVER_H
#define KKTSOLVER_H

#include "initCompliant.h"

#include "AssembledSystem.h"
#include <sofa/core/objectmodel/BaseObject.h>


namespace sofa {
namespace component {
namespace linearsolver {
			
// Solver for an AssembledSystem (could be non-linear in case of
// inequalities). This will eventually serve as a base class for
// all kinds of derived solver (sparse cholesky, minres, qp)
			
// TODO: base + derived classes (minres/cholesky/unilateral)
class SOFA_Compliant_API KKTSolver : public virtual core::objectmodel::BaseObject {
  public:
	SOFA_CLASS(KKTSolver, core::objectmodel::BaseObject);

	typedef AssembledSystem::vec vec;

	// solve the KKT system: \mat{ M - h^2 K & J^T \\ J, -C } x = rhs
	// (watch out for the compliance scaling)
	
	virtual void factor(const AssembledSystem& system) = 0;
	
	virtual void solve(vec& x,
	                   const AssembledSystem& system,
	                   const vec& rhs) const = 0;
	
};


}
}
}

#endif
