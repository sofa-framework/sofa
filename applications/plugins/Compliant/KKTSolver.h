#ifndef KKTSOLVER_H
#define KKTSOLVER_H

#include "initCompliant.h"

#include "AssembledSystem.h"
#include <sofa/core/behavior/LinearSolver.h>

namespace sofa {
namespace component {
namespace linearsolver {
			
// Solver for an AssembledSystem (could be non-linear in case of
// inequalities). This will eventually serve as a base class for
// all kinds of derived solver (sparse cholesky, minres, qp)





			
// TODO: base + derived classes (minres/cholesky/unilateral)
class SOFA_Compliant_API KKTSolver : public core::behavior::BaseLinearSolver {
  public:
	SOFA_CLASS(KKTSolver, core::objectmodel::BaseObject);

	// solve the KKT system: \mat{ M - h^2 K & J^T \\ J, -C } x = rhs
	// (watch out for the compliance scaling)
	
	typedef AssembledSystem system_type;
	typedef system_type::vec vec;

	virtual void factor(const system_type& system) = 0;
	
	virtual void solve(vec& x,
	                   const system_type& system,
                       const vec& rhs) const = 0;

    // optional constraint filter
    // TODO take into account this mask in every solver
    // WARNING this active mask cannot be considered by every solvers (LDLT for instance)
    mutable vec _constraintMask;


    // return true if the solver can only handle equality constraints (in opposition with LCP for instance)
    virtual bool isLinear() const { return true; }
};


}
}
}

#endif
