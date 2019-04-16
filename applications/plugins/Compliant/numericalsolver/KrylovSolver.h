#ifndef COMPLIANT_KRYLOVSOLVER_H
#define COMPLIANT_KRYLOVSOLVER_H

#include "IterativeSolver.h"
#include "Response.h"

#include "../utils/krylov.h"

#include "../assembly/AssembledSystem.h"
#include <sofa/core/objectmodel/BaseObject.h>

#include "SubKKT.h"

template<class> struct schur;
struct kkt;

namespace sofa {
namespace component {
namespace linearsolver {


/// Base class for iterative solvers in the Krylov family (CG, Minres, ...)
class SOFA_Compliant_API KrylovSolver : public IterativeSolver {
  public:
	
	KrylovSolver();				
	
	Data<bool> verbose; ///< print debug stuff on std::cerr
    Data<unsigned> restart; ///< restart every n steps
	
	void init() override;
	
	void solve(vec& x,
	                   const system_type& system,
                       const vec& rhs) const override;

	void correct(vec& x,
						 const system_type& system,
						 const vec& rhs,
						 real damping) const override;


	void factor(const system_type& sys) override;
	
  protected:

	virtual void solve_schur(vec& x,
	                         const system_type& system,
	                         const vec& rhs, 
							 real damping = 0) const;

    
	virtual void solve_kkt(vec& x,
	                       const system_type& system,
	                       const vec& rhs,
						   real damping = 0) const;
	
	typedef ::krylov<SReal>::params params_type;

	// convenience
	virtual params_type params(const vec& rhs) const;

	// again
    void report(const params_type& p) const;


    typedef ::schur<SubKKT::Adaptor> schur_type;
    virtual void solve_schur_impl(vec& lambda,
                                  const schur_type& A,
                                  const vec& b,
                                  params_type& p) const = 0;

    typedef ::kkt kkt_type;
    virtual void solve_kkt_impl(vec& x,
                                const kkt_type& A,
                                const vec& b,
                                params_type& p) const = 0;

    virtual const char* method() const = 0;
    
	typedef Response response_type;
	Response::SPtr response;
    SubKKT sub;

    
public:

  Data<bool> parallel; ///< use openmp to parallelize matrix-vector products when use_schur is false (parallelization per KKT blocks, 4 threads)

};


}
}
}

#endif
