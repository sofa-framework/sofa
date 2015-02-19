#ifndef COMPLIANT_NUMSOLVER_KKTSOLVER_H
#define COMPLIANT_NUMSOLVER_KKTSOLVER_H

#include <Compliant/Compliant.h>
#include "../assembly/AssembledSystem.h"

#include <sofa/core/behavior/LinearSolver.h>


namespace sofa {
namespace component {
namespace linearsolver {

/// Base class to solve a KKT system (Karush–Kuhn–Tucker conditions)
/// Solver for an AssembledSystem (could be non-linear in case of
/// inequalities). This will eventually serve as a base class for
/// all kinds of derived solver (sparse cholesky, minres, qp)
class SOFA_Compliant_API KKTSolver : public core::behavior::BaseLinearSolver {
  public:
	SOFA_CLASS(KKTSolver, core::objectmodel::BaseObject);

	// solve the KKT system: \mat{ M - h^2 K & J^T \\ J, -C } x = rhs
	// (watch out for the compliance scaling)
	
	typedef AssembledSystem system_type;
	
	typedef system_type::real real;
    typedef system_type::vec vec;
    typedef system_type::mat mat;

    Data<bool> debug; ///< print debug info

    KKTSolver() : debug(initData(&debug,false,"debug","print debug info")) {}

	
	virtual void factor(const system_type& system) = 0;
	
	virtual void solve(vec& x,
	                   const system_type& system,
                       const vec& rhs) const = 0;


	// performs a correction pass, by default the same as dynamics
	// unless derived classes know better (e.g. only correct normal
	// constraints for friction)
	// damping allows to numerically damp unfeasible problems that may
	// arise during correction
	virtual void correct(vec& x,
						 const system_type& system,
						 const vec& rhs,
                         real /*damping*/ = 0) const {
		solve(x, system, rhs);
	}



};


}
}
}

#endif
