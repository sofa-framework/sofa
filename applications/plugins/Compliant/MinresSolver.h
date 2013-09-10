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

	// solve the KKT system: \mat{ M - h^2 K & J^T \\ J, -C } x = rhs
	// (watch out for the compliance scaling)
	virtual void solve(vec& x,
	                   const system_type& system,
	                   const vec& rhs) const;

	virtual void factor(const AssembledSystem& system);
				
  protected:
	// virtual void solve_schur(vec& x,
	//                          const system_type& system,
	//                          const vec& rhs) const;
	
	virtual void solve_kkt(vec& x,
	                       const system_type& system,
	                       const vec& rhs) const;
	
	
  public:
	// Data<bool> use_schur, fast_schur;
	
	Data<bool> parallel;
	
	Data<bool> verbose;

  protected:
	
	// typedef system_type::real real;
	// typedef Eigen::SparseMatrix<real, Eigen::ColMajor> cmat;
	// typedef Eigen::SimplicialLDLT< cmat > response_type;
	
	// response matrix for schur solves
	// response_type response;
};


}
}
}

#endif
