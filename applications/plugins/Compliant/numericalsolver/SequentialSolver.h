#ifndef COMPLIANT_SEQUENTIALSOLVER_H
#define COMPLIANT_SEQUENTIALSOLVER_H


// #include "utils/debug.h"
#include "../initCompliant.h"

#include "KKTSolver.h"
#include "Response.h"

#include "../constraint/Constraint.h"


#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/Cholesky>

#include <boost/function.hpp>

namespace sofa {
namespace component {
namespace linearsolver {

// sequential impulse/projected block gauss-seidel kkt solver
class SOFA_Compliant_API SequentialSolver : public KKTSolver {
  public:

	SOFA_CLASS(SequentialSolver, KKTSolver);
	
	SequentialSolver();

	virtual void factor(const system_type& system);
	
	virtual void solve(vec& x,
	                   const system_type& system,
                       const vec& rhs) const;

	virtual bool isLinear() const { return false; }
	
	virtual void init();

    virtual void setCorrectionPass( bool b  ){ _correctionPass=b; }
	
	Data<unsigned> iterations;
	Data<SReal> precision;
	Data<bool> relative;
	Data<SReal> omega;

	Data<std::string> bench;

    Data<bool> projectH;    ///< Replace H with P^T.H.P to account for projective constraints

  protected:

    bool _correctionPass;

	// performs a single iteration
	SReal step(vec& lambda, 
	           vec& net, 
	           const system_type& sys,
	           const vec& rhs,
	           vec& tmp1, vec& tmp2 ) const;
	
	// response matrix
	Response::SPtr response;
	
	// mapping matrix response 
    typedef Response::cmat cmat;
    cmat mapping_response;

	// data blocks 
	struct block {
		block();
		unsigned offset, size;
        Constraint* projector;
	};
	
	typedef std::vector<block> blocks_type;
	blocks_type blocks;
	
	void fetch_blocks(const system_type& system);

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
	void solve_block(chunk_type result, const inverse_type& inv, chunk_type rhs) const;
	
  protected:


};

}
}
}

#endif
