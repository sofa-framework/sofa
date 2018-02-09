#ifndef COMPLIANT_EIGENSPARSERESPONSE_H
#define COMPLIANT_EIGENSPARSERESPONSE_H

#include "Response.h"
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

namespace sofa {
namespace component {
namespace linearsolver {

/// Solving the dynamics equation
/// @warning with "constant" set to true, mass and stiffness cannot be added dynamically (like a mouse-spring or penalities)
/// @author Matthieu Nesme
template<class LinearSolver, bool symmetric=false>
class SOFA_Compliant_API EigenSparseResponse : public Response {
public:
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE2(EigenSparseResponse,LinearSolver,symmetric), Response);

    EigenSparseResponse();

    virtual void factor(const rmat& sys);
	virtual void solve(cmat& lval, const cmat& rval) const;
    virtual void solve(vec& lval, const vec& rval) const;
    virtual void reinit();
    virtual bool isSymmetric() const { return symmetric; }

    /// Add identity*regularize to matrix H to make it definite (Tikhonov regularization)
    /// (this is useful when H is projected with a projective constraint and becomes semidefinite)
    Data<SReal> d_regularize;

	/// reuse first factorization
    Data<bool> d_constant;

    /// if the sparsity pattern remains similar from one step to the other,
    /// the factorization can be faster
    Data<bool> d_trackSparsityPattern;

protected:

	typedef system_type::real real;
    typedef LinearSolver response_type;
	
	response_type response;

    bool m_factorized;
    cmat::Index m_previousSize; ///< @internal for d_trackSparsityPattern
    cmat::Index m_previousNonZeros; ///< @internal for d_trackSparsityPattern

    cmat tmp;

    /// @internal perform the factorization
    void compute( const cmat& M );
	
};



/// Solving the dynamics equation using a LDL^T Cholesky decomposition
/// Working for any symmetric positive (semi)definite matrix
/// @warning this sparse implementation from Eigen does not handle semidefinite matrices (only definite ones)
/// (basically for any regular mechanical systems including rigids and deformables,
/// it is why it is used by default)
typedef Eigen::SimplicialLDLT< AssembledSystem::cmat > LDLTSparseLinearSolver;
class SOFA_Compliant_API LDLTResponse : public EigenSparseResponse< LDLTSparseLinearSolver,true>
{
public:
    SOFA_CLASS(LDLTResponse,SOFA_TEMPLATE2(EigenSparseResponse,LDLTSparseLinearSolver,true));
};



/// Solving the dynamics equation using a LLT Cholesky decomposition
/// Working for any symmetric positive definite matrix
/// (basically for any regular mechanical systems including rigids and deformables,
/// it is why it is used by default)
/// It does not handle semidefinite matrices but should be more efficient than LDLT
typedef Eigen::SimplicialLLT< AssembledSystem::cmat > LLTSparseLinearSolver;
class SOFA_Compliant_API LLTResponse : public EigenSparseResponse< LLTSparseLinearSolver,true>
{
public:
    SOFA_CLASS(LLTResponse,SOFA_TEMPLATE2(EigenSparseResponse,LLTSparseLinearSolver,true));
};



/// Solving the dynamics equation using a LU factorization
/// Working for any square matrix (particularly useful for non-symmetric matrices, otherwise prefer the LDLTResponse)
typedef Eigen::SparseLU< AssembledSystem::cmat > LUSparseLinearSolver;
class SOFA_Compliant_API LUResponse : public EigenSparseResponse< LUSparseLinearSolver >
{
public:
    SOFA_CLASS(LUResponse,SOFA_TEMPLATE(EigenSparseResponse,LUSparseLinearSolver));
};


/////////////////////



/// Solve a dynamics system including bilateral constraints with an iterative linear solver
template<class LinearSolver, bool symmetric=false>
class EigenSparseIterativeResponse : public EigenSparseResponse< LinearSolver, symmetric >
{
public:

    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE2(EigenSparseIterativeResponse,LinearSolver,symmetric),SOFA_TEMPLATE2(EigenSparseResponse,LinearSolver,symmetric));

    Data<unsigned> d_iterations; ///< max iterations
    Data<SReal> d_tolerance; ///< tolerance

    EigenSparseIterativeResponse();

    virtual void init();
    virtual void reinit();

};

typedef Eigen::ConjugateGradient< AssembledSystem::cmat > CGSparseLinearSolver;
class EigenCGResponse : public EigenSparseIterativeResponse< CGSparseLinearSolver, true >
{
public:
    SOFA_CLASS(EigenCGResponse,SOFA_TEMPLATE2(EigenSparseIterativeResponse,CGSparseLinearSolver, true));
};


}
}
}

#endif
