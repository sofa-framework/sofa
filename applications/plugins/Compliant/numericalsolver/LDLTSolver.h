#ifndef LDLTSOLVER_H
#define LDLTSOLVER_H

#include "KKTSolver.h"


#include <Eigen/LU>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <Eigen/SVD>

#include "utils/thread_local.h"


namespace sofa {
namespace component {
namespace linearsolver {

// 
class SOFA_Compliant_API LDLTSolver : public KKTSolver {
  public:
	
	SOFA_CLASS(LDLTSolver, KKTSolver);
	
	virtual void solve(vec& x,
	                   const AssembledSystem& system,
	                   const vec& rhs) const;

	// performs factorization
	virtual void factor(const AssembledSystem& system);

	LDLTSolver();
	~LDLTSolver();


    Data<bool> projectH;    ///< Replace H with P^T.H.P to account for projective constraints
    Data<SReal> regularize; ///< Add identity*regularize to matrix H to make it definite. This is useful when H is projected (see @projectH).

  private:
	
	struct pimpl_type {
		typedef SReal real;
		typedef Eigen::SparseMatrix<real, Eigen::ColMajor> cmat;
		typedef Eigen::SparseMatrix<real, Eigen::RowMajor> rmat;
		
		typedef Eigen::SimplicialLDLT< cmat >  solver_type;
		
		unsigned m, n;
		solver_type Hinv, schur;
		cmat H, HinvPJT;
		SReal dt;
		
	};

	
	mutable thread_local<pimpl_type> pimpl;

};


}
}
}



#endif
