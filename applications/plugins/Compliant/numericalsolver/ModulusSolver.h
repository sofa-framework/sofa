#ifndef COMPLIANT_MODULUSSOLVER_H
#define COMPLIANT_MODULUSSOLVER_H

#include "IterativeSolver.h"
#include "../utils/sub_kkt.h"

#include <Eigen/SparseCholesky>

namespace sofa {
namespace component {
namespace linearsolver {

// solve unilateral sparse systems without forming the schur
// complement
class SOFA_Compliant_API ModulusSolver : public IterativeSolver {
  public:
	
	SOFA_CLASS(ModulusSolver, IterativeSolver);
	
	virtual void solve(vec& x,
	                   const system_type& sys,
	                   const vec& rhs) const;

	virtual void factor(const system_type& sys);

	ModulusSolver();
	~ModulusSolver();
    
  protected:

    class sub_kkt : public utils::sub_kkt {
    public:
        using utils::sub_kkt::matrix;
        using utils::sub_kkt::primal;        
    };
    
    sub_kkt sub;

    typedef Eigen::SimplicialLDLT<cmat, Eigen::Upper> solver_type;
    solver_type solver;
    
    Data<real> omega;
    Data<unsigned> anderson;
    
  private:
    vec unilateral, diagonal;
    
};


}
}
}



#endif
