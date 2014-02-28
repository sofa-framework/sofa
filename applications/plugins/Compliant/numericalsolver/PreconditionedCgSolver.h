#ifndef COMPLIANT_PRECONDITIONEDCGSOLVER_H
#define COMPLIANT_PRECONDITIONEDCGSOLVER_H

#include "CgSolver.h"
#include "PreconditionedSolver.h"

namespace sofa {
namespace component {
namespace linearsolver {


class SOFA_Compliant_API PreconditionedCgSolver : public CgSolver, public PreconditionedSolver
{

  public:
    SOFA_CLASS(PreconditionedCgSolver, CgSolver);

    virtual void init();

  protected:

	
	virtual void solve_kkt(vec& x,
	                       const system_type& system,
	                       const vec& rhs,
						   real damping) const;

};


}
}
}

#endif
