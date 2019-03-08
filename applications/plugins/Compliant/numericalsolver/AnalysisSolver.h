#ifndef COMPLIANT_ANALYSISSOLVER_H
#define COMPLIANT_ANALYSISSOLVER_H


#include <Compliant/config.h>

#include "KKTSolver.h"

namespace sofa {
namespace component {
namespace linearsolver {

/// simple solver that discovers kktsolvers at its level, then launch
/// them sequentially on each kkt problem
class SOFA_Compliant_API AnalysisSolver : public KKTSolver {
  public:

	SOFA_CLASS(AnalysisSolver, KKTSolver);
	 
	AnalysisSolver();
	
	void init() override;

	void factor(const system_type& system) override;
	
	void solve(vec& x,
	                   const system_type& system,
	                   const vec& rhs) const override;

	void correct(vec& x,
						 const system_type& system,
						 const vec& rhs, real damping) const override;

    Data<bool> condest; ///< estimating the condition number as a ratio largest/smaller singular values (computed by SVD)
    Data<bool> eigenvaluesign; ///< computing the sign of the eigenvalues (of the implicit matrix H)

    Data<std::string> dump_qp;       ///< dump qp to given filename if non-empty

  protected:

    std::vector< KKTSolver::SPtr > solvers;

};


}
}
}

#endif
