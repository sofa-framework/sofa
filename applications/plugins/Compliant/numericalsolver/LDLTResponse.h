#ifndef COMPLIANT_LDLTRESPONSE_H
#define COMPLIANT_LDLTRESPONSE_H

#include "Response.h"
#include <Eigen/SparseCholesky>

namespace sofa {
namespace component {
namespace linearsolver {

/// Solving the dynamics equation using a LDL^T Cholesky decomposition
///
/// Working for any symmetric positive semidefinite matrix
/// (basically for any regular mechanical systems including rigids and deformables,
/// it is why it is used by default)
///
/// Note that more optimized Response can be used:
/// - for a scene including only Rigids, when the dynamics matrix includes only a constant diagonal mass/inertia
///     -> prefer a constant DiagonalResponse
/// - for a scene including deformables with a constant stiffness (linear stiffness or simulated in compliance)
///     -> set "constant" to true
/// - for a scene with regular deformables (non constant stiffness or even non constant mass)
///     -> LDLTResponse is the right component!
///
/// @warning with "constant" set to true, mass and stiffness cannot be added dynamically (like a mouse-spring or penalities)
class SOFA_Compliant_API LDLTResponse : public Response {
public:
	SOFA_CLASS(LDLTResponse, Response);

    LDLTResponse();

    virtual void factor(const mat& sys, bool semidefinite=false);
	virtual void solve(cmat& lval, const cmat& rval) const;
	virtual void solve(vec& lval,  const vec& rval) const;
    virtual void reinit();

    /// Add identity*regularize to matrix H to make it definite (Tikhonov regularization)
    /// (this is useful when H is projected with a projective constraint and becomes semidefinite)
    Data<SReal> regularize;

	/// reuse first factorization
	Data<bool> constant;

protected:

	typedef system_type::real real;
	typedef Eigen::SimplicialLDLT< cmat > response_type;
	
	response_type response;

    bool factorized;
	
};

}
}
}

#endif
