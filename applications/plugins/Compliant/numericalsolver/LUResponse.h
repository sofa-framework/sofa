#ifndef COMPLIANT_LUResponse_H
#define COMPLIANT_LUResponse_H

#include "Response.h"
#include <Eigen/SparseLU>

namespace sofa {
namespace component {
namespace linearsolver {

/// Solving the dynamics equation using a LU factorization
///
/// Working for any square matrix (particularly useful for non-symmetric matrices, otherwise prefer the LDLTResponse)
class SOFA_Compliant_API LUResponse : public Response {
public:
    SOFA_CLASS(LUResponse, Response);

    LUResponse();

    virtual void factor(const mat& sys, bool semidefinite=false);
	virtual void solve(cmat& lval, const cmat& rval) const;
    virtual void solve(vec& lval,  const vec& rval) const;

    /// Add identity*regularize to matrix H to make it definite (Tikhonov regularization)
    /// (this is useful when H is projected with a projective constraint and becomes semidefinite)
    Data<SReal> regularize;

protected:

	typedef system_type::real real;
    typedef Eigen::SparseLU< cmat > response_type;
	
    response_type response;
	
};

}
}
}

#endif
