#ifndef COMPLIANT_EIGENSPARSERESPONSE_H
#define COMPLIANT_EIGENSPARSERESPONSE_H

#include "Response.h"
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

namespace sofa {
namespace component {
namespace linearsolver {

/// Solving the dynamics equation
/// @warning with "constant" set to true, mass and stiffness cannot be added dynamically (like a mouse-spring or penalities)
template<class LinearSolver, int symmetric>
class SOFA_Compliant_API EigenSparseResponse : public Response {
public:
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE2(EigenSparseResponse,LinearSolver,symmetric), Response);

    EigenSparseResponse();

    virtual void factor(const mat& sys, bool semidefinite=false);
	virtual void solve(cmat& lval, const cmat& rval) const;
	virtual void solve(vec& lval,  const vec& rval) const;
    virtual void reinit();

    /// Add identity*regularize to matrix H to make it definite (Tikhonov regularization)
    /// (this is useful when H is projected with a projective constraint and becomes semidefinite)
    Data<SReal> d_regularize;

	/// reuse first factorization
    Data<bool> d_constant;

protected:

	typedef system_type::real real;
    typedef LinearSolver response_type;
	
	response_type response;

    bool m_factorized;
	
};



/// Solving the dynamics equation using a LDL^T Cholesky decomposition
/// Working for any symmetric positive semidefinite matrix
/// (basically for any regular mechanical systems including rigids and deformables,
/// it is why it is used by default)
typedef Eigen::SimplicialLDLT< AssembledSystem::cmat > LDLTSparseLinearSolver;
class SOFA_Compliant_API LDLTResponse : public EigenSparseResponse< LDLTSparseLinearSolver, LDLTSparseLinearSolver::UpLo >
{
public:
    SOFA_CLASS(LDLTResponse,SOFA_TEMPLATE2(EigenSparseResponse,LDLTSparseLinearSolver,LDLTSparseLinearSolver::UpLo));
};


/// Solving the dynamics equation using a LU factorization
/// Working for any square matrix (particularly useful for non-symmetric matrices, otherwise prefer the LDLTResponse)
typedef Eigen::SparseLU< AssembledSystem::cmat > LUSparseLinearSolver;
class SOFA_Compliant_API LUResponse : public EigenSparseResponse< LUSparseLinearSolver, 0 >
{
public:
    SOFA_CLASS(LUResponse,SOFA_TEMPLATE2(EigenSparseResponse,LUSparseLinearSolver,0));
};




}
}
}

#endif
