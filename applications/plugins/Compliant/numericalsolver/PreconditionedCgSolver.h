#ifndef COMPLIANT_PRECONDITIONEDCGSOLVER_H
#define COMPLIANT_PRECONDITIONEDCGSOLVER_H

#include "CgSolver.h"
#include "PreconditionedSolver.h"

namespace sofa {
namespace component {
namespace linearsolver {

/// Preconditioned Conjugate Gradient (iterative, linear solver for symmetric, definite matrix)
///
/// @author Matthieu Nesme
///
class SOFA_Compliant_API PreconditionedCgSolver : public CgSolver, public PreconditionedSolver
{

  public:
    SOFA_CLASS(PreconditionedCgSolver, CgSolver);

    void init() override;

  protected:

	
	void solve_kkt(vec& x,
	                       const system_type& system,
	                       const vec& rhs,
						   real damping) const override;

    const char* method() const override;
};


}
}
}

#endif
