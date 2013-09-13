#ifndef COMPLIANT_MINRESSOLVER_H
#define COMPLIANT_MINRESSOLVER_H

#include "KrylovSolver.h"

#include <Eigen/SparseCholesky>

namespace sofa {
namespace component {
namespace linearsolver {
			
// TODO add back schur solves
class SOFA_Compliant_API MinresSolver : public KrylovSolver {
  public:
	SOFA_CLASS(MinresSolver, KrylovSolver);
	
	MinresSolver();				
	
	typedef AssembledSystem system_type;
	typedef system_type::vec vec;

  protected:

	virtual void solve_schur(vec& x,
	                         const system_type& system,
	                         const vec& rhs) const;
	
	virtual void solve_kkt(vec& x,
	                       const system_type& system,
	                       const vec& rhs) const;
	
	
  public:
	
	Data<bool> parallel;
	
  protected:
	
};


}
}
}

#endif
