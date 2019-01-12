#ifndef COMPLIANT_NNCGSolver_H
#define COMPLIANT_NNCGSolver_H

#include <Compliant/config.h>
#include "SequentialSolver.h"

namespace sofa {
namespace component {
namespace linearsolver {

/// Sequential impulse/projected block gauss-seidel kkt solver with post-iteration conjugation
/// Morten Silcowitz-Hansen, Sarah Niebe and Kenny Erleben, Nonsmooth Nonlinear conjugate gradient solver, 2010
/// @author Maxime Tournier
class SOFA_Compliant_API NNCGSolver : public SequentialSolver {
  public:

    SOFA_CLASS(NNCGSolver, SequentialSolver);
	
    NNCGSolver();
	
  protected:
	virtual void solve_impl(vec& x,
							const system_type& system,
							const vec& rhs, 
							bool correct,
                            real damping) const;

	
	Data<bool> verbose; ///< print stuff
};

}
}
}

#endif
