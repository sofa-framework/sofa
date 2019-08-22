#ifndef COMPLIANT_UTILS_SUB_KKT_H
#define COMPLIANT_UTILS_SUB_KKT_H

// please don't touch this one, derive from it instead

#include "../utils/eigen_types.h"

// forward decls
namespace sofa {
namespace component {
namespace linearsolver {

  class AssembledSystem;

}
}
}

namespace utils {

  
  class sub_kkt : public eigen_types {

  protected:
	// primal/dual selection matrices
    rmat primal, dual;

	// projected subsystem
	rmat matrix;

	struct helper;
	
  private:
	mutable vec vtmp1, vtmp2;
	mutable cmat cmtmp1, cmtmp2;

  public:

	typedef sofa::component::linearsolver::AssembledSystem system_type;

	// standard projected (1, 1) schur subsystem (dual is
    // empty). size_full = sys.m, size_sub = #(non-zero sys.P elements)
    void projected_primal(const system_type& sys);

	// full kkt with projected primal variables
    // @param eps adds compliance
    // only_lower to build only the low triangular matrix for symmetric problems
    void projected_kkt(const system_type& sys,
					   real eps = 0,
					   bool only_lower = false);

	// you may specialize this one to adapt to different API (Response
	// by default)
	template<class Solver>
	struct traits;

	template<class Solver>
	void factor(Solver& solver) const;

	template<class Solver>
	void solve(const Solver& solver, vec& res, const vec& rhs) const;

	// // TODO is this one even needed ?
	// template<class Solver>
	// void solve(const Solver& solver, cmat& res, const cmat& rhs) const;
	
	void prod(vec& res, const vec& rhs, bool only_lower = false) const;
	
  protected:

	// primal.rows() + dual.rows()
    unsigned size_full() const;

    // primal.cols() + dual.cols()
    unsigned size_sub() const;

	// project rhs, do stuff, unproject result
	template<class Action>
    void project_unproject(const Action& action, vec& res, const vec& rhs) const;

	template<class Solver> struct solve_action;
	struct prod_action;
	
  };

  

}


#endif
