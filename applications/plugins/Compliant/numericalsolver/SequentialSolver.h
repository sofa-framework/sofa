#ifndef COMPLIANT_SEQUENTIALSOLVER_H
#define COMPLIANT_SEQUENTIALSOLVER_H


// #include "utils/debug.h"
#include "../initCompliant.h"

#include "IterativeSolver.h"
#include "Response.h"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/Cholesky>

namespace sofa {
namespace component {
namespace linearsolver {

/// Sequential impulse/projected block gauss-seidel kkt solver
class SOFA_Compliant_API SequentialSolver : public IterativeSolver {
  public:

	SOFA_CLASS(SequentialSolver, IterativeSolver);
	
	SequentialSolver();

	virtual void factor(const system_type& system);
	
	virtual void solve(vec& x,
	                   const system_type& system,
                       const vec& rhs) const;

	virtual void correct(vec& x,
						 const system_type& system,
						 const vec& rhs,
						 real damping) const;

	virtual void init();

	Data<SReal> omega;

  protected:

	virtual void solve_impl(vec& x,
							const system_type& system,
							const vec& rhs,
							bool correct) const;

    // performs a single iteration
    SReal step(vec& lambda,
	           vec& net, 
	           const system_type& sys,
	           const vec& rhs,
	           vec& tmp1, vec& tmp2,
			   bool correct = false) const;
	
	// response matrix
	typedef Response response_type;
	response_type::SPtr response;
	
	// mapping matrix response 
    typedef Response::cmat cmat;
    cmat mapping_response;
	mat JP;
	
	// data blocks 
	struct SOFA_Compliant_API block {
		block();
        unsigned offset, size;
        Constraint* projector;
        bool activated; // is the constraint activated, otherwise its lambda is forced to be 0
	};
	
	typedef std::vector<block> blocks_type;
	blocks_type blocks;
	
    virtual void fetch_blocks(const system_type& system);

	// constraint responses
	typedef Eigen::Matrix< system_type::real, Eigen::Dynamic, Eigen::Dynamic > dense_matrix;
	typedef Eigen::LDLT< dense_matrix > inverse_type;

	// blocks inverse
	typedef std::vector< inverse_type > blocks_inv_type;
	blocks_inv_type blocks_inv;
	
	// blocks factorization
	typedef Eigen::Map<dense_matrix> schur_type;
	void factor_block(inverse_type& inv, const schur_type& schur);
	
	// blocks solve
	typedef Eigen::Map< vec > chunk_type;
	virtual void solve_block(chunk_type result, const inverse_type& inv, chunk_type rhs) const;
	
  protected:


};

}
}
}

#endif
