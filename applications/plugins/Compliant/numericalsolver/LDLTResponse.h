#ifndef COMPLIANT_LDLTRESPONSE_H
#define COMPLIANT_LDLTRESPONSE_H

#include "Response.h"
#include <Eigen/SparseCholesky>

namespace sofa {
namespace component {
namespace linearsolver {

class LDLTResponse : public Response {
public:
	SOFA_CLASS(LDLTResponse, Response);

    LDLTResponse();

	virtual void factor(const mat& sys);
	virtual void solve(cmat& lval, const cmat& rval) const;
	virtual void solve(vec& lval,  const vec& rval) const;

    Data<SReal> regularize; ///< Add identity*regularize to matrix H to make it definite (this is useful with a projective contraint)

protected:

	typedef system_type::real real;
	typedef Eigen::SimplicialLDLT< cmat > response_type;
	
	response_type response;
	
};

}
}
}

#endif
