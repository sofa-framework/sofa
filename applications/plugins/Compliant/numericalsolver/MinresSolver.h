#ifndef COMPLIANT_MINRESSOLVER_H
#define COMPLIANT_MINRESSOLVER_H

#include "KrylovSolver.h"

namespace sofa {
namespace component {
namespace linearsolver {
			
/// Minimal Residual Method (iterative, linear solver for symmetric, semidefinite matrix)
// TODO add back schur solves
class SOFA_Compliant_API MinresSolver : public KrylovSolver {
  public:
	SOFA_CLASS(MinresSolver, KrylovSolver);

	typedef AssembledSystem system_type;
	typedef system_type::vec vec;

  protected:

    void solve_schur_impl(vec& lambda,
                                  const schur_type& A,
                                  const vec& b,
                                  params_type& p) const override;

    void solve_kkt_impl(vec& x,
                                const kkt_type& A,
                                const vec& b,
                                params_type& p) const override;

    const char* method() const override;
    
};


}
}
}

#endif
