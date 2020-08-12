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
	
	void solve(vec& x,
	                   const system_type& sys,
	                   const vec& rhs) const override;

	void factor(const system_type& sys) override;

	ModulusSolver();
	~ModulusSolver() override;
    
  protected:

    class sub_kkt : public utils::sub_kkt {
    public:
        using utils::sub_kkt::matrix;
        using utils::sub_kkt::primal;        
    };


    void project(vec::SegmentReturnType view, const system_type& sys, bool correct) const;
    
    sub_kkt sub;

    typedef Eigen::SimplicialLDLT<cmat, Eigen::Upper> solver_type;
    solver_type solver;
    
    Data<real> omega;
    
    Data<unsigned> anderson;
    Data<bool> nlnscg;
    
  private:
    vec unilateral, diagonal;
    
};


}
}
}



#endif
