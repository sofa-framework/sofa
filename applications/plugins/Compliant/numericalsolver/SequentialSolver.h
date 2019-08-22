#ifndef COMPLIANT_BaseSequentialSolver_H
#define COMPLIANT_BaseSequentialSolver_H


// #include "utils/debug.h"
#include <Compliant/config.h>

#include "IterativeSolver.h"
#include "Response.h"
#include "SubKKT.h"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/Cholesky>

namespace sofa {
namespace component {
namespace linearsolver {

/// Sequential impulse/projected block gauss-seidel kkt solver
class SOFA_Compliant_API BaseSequentialSolver : public IterativeSolver {
  public:

    SOFA_ABSTRACT_CLASS(BaseSequentialSolver, IterativeSolver);
	
    BaseSequentialSolver();

	void factor(const system_type& system) override;

    void solve(vec& x,
                       const system_type& system,
                       const vec& rhs) const override;

    void correct(vec& x,
                         const system_type& system,
                         const vec& rhs,
                         real damping) const override;

	void init() override;

    Data<SReal> omega; ///< SOR parameter:  omega < 1 : better, slower convergence, omega = 1 : vanilla gauss-seidel, 2 > omega > 1 : faster convergence, ok for SPD systems, omega > 2 : will probably explode

  protected:

	virtual void solve_impl(vec& x,
							const system_type& system,
							const vec& rhs,
                            bool correct,
                            real damping = 0) const;

    virtual void factor_impl(const system_type& system);


    // performs a single iteration
    SReal step(vec& lambda,
	           vec& net, 
	           const system_type& sys,
	           const vec& rhs,
	           vec& tmp1, vec& tmp2,
			   bool correct = false,
               real damping = 0) const;
	
	// response matrix
	typedef Response response_type;
    response_type::SPtr response;
    SubKKT sub;
	
	// mapping matrix response 
    typedef Response::cmat cmat;
    cmat mapping_response;
	rmat JP;
	
	// data blocks 
	struct SOFA_Compliant_API block {
		block();
        unsigned offset, size;
        Constraint* projector;
        bool activated; // is the constraint activated, otherwise its lambda is forced to be 0
	};
	
	typedef std::vector<block> blocks_type;
	blocks_type blocks;

    void fetch_blocks(const system_type& system);

    // constraint responses
    typedef Eigen::LDLT< dmat > inverse_type;

	// blocks inverse
	typedef std::vector< inverse_type > blocks_inv_type;
	blocks_inv_type blocks_inv;
	
	// blocks factorization
    typedef Eigen::Map<dmat> schur_type;
	void factor_block(inverse_type& inv, const schur_type& schur);

	// blocks solve
	typedef Eigen::Map< vec > chunk_type;
    void solve_block(chunk_type result, const inverse_type& inv, chunk_type rhs) const;



};



/// Projected block gauss-seidel kkt solver with schur complement on both dynamics+bilateral constraints
/// TODO move this as an option in BaseSequentialSolver, so it can be used by derived solvers (eg NNCGSolver)
class SOFA_Compliant_API SequentialSolver : public BaseSequentialSolver {

protected:

    // let's play with matrices
    struct LocalSubKKT : public SubKKT
    {
        // kkt with projected primal variables and bilateral constraints
        // excludes non bilateral constraints
        bool projected_primal_and_bilateral( AssembledSystem& res,
                                  const AssembledSystem& sys,
                                  real eps = 0,
                                  bool only_lower = false);


        // reorder global (primal,constraint) to projected (primal,bilateral,non-bilateral) order
        void toLocal(vec& local, const vec& global ) const;
        // reorder projected (primal,bilateral,non-bilateral) to global (primal,constraint) order
        void fromLocal( vec& global, const vec& local ) const;

    protected:

        // non-bilateral constraints selection matrix
        rmat Q_unil;
    };


public:

    SOFA_CLASS(SequentialSolver, BaseSequentialSolver);

    SequentialSolver();

    Data<bool> d_iterateOnBilaterals; ///< Should the bilateral constraint must be solved iteratively or factorized with the dynamics?
    Data<SReal> d_regularization; ///< Optional diagonal Tikhonov regularization on bilateral constraints

    void factor(const system_type& system) override;

    void solve(vec& x,
                       const system_type& system,
                       const vec& rhs) const override;

    void correct(vec& x,
                         const system_type& system,
                         const vec& rhs,
                         real damping) const override;

protected:


    system_type m_localSystem; // a local AssembledSystem with H=(primal,bilateral), J=J_unil, P=identity (H already filtered)
    LocalSubKKT m_localSub; // to build m_localSystem


    void solve_local(vec& x,
                     const system_type& system,
                     const vec& rhs,
                     bool correct,
                     real damping = 0) const;

    virtual void fetch_unilateral_blocks(const system_type& system);

};



}
}
}

#endif
