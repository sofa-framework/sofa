#ifndef COMPLIANT_CGSOLVER_H
#define COMPLIANT_CGSOLVER_H

#include "KrylovSolver.h"


namespace sofa {
namespace component {
namespace linearsolver {
			

class SOFA_Compliant_API CgSolver : public KrylovSolver {

  public:
	SOFA_CLASS(CgSolver, KrylovSolver);
	
	CgSolver();				
	
	typedef AssembledSystem system_type;
	typedef system_type::vec vec;
	
	// solve the KKT system: \mat{ M - h^2 K & J^T \\ J, -C } x = rhs
	// (watch out for the compliance scaling)
	virtual void solve(vec& x,
	                   const system_type& system,
	                   const vec& rhs) const;

	virtual void factor(const AssembledSystem& system);
	

  protected:
	
	typedef system_type::real real;
	typedef Eigen::SparseMatrix<real, Eigen::ColMajor> cmat;
	typedef Eigen::SimplicialLDLT< cmat > response_type;
	
};


}
}
}

#endif
