#ifndef COMPLIANT_DIAGONALRESPONSE_H
#define COMPLIANT_DIAGONALRESPONSE_H

#include "initCompliant.h"
#include "Response.h"

namespace sofa {
namespace component {
namespace linearsolver {

class SOFA_Compliant_API DiagonalResponse : public Response {
public:
	SOFA_CLASS(DiagonalResponse, Response);
	
	DiagonalResponse();

	virtual void factor(const mat& sys);
	virtual void solve(cmat& lval, const cmat& rval) const;
	virtual void solve(vec& lval,  const vec& rval) const;

	const vec& diagonal() const { return diag; }

    /// Add identity*regularize to matrix H to make it definite (this
    /// is useful with a projective contraint)
    Data<SReal> regularize;

    Data<bool> constant;
	
protected:

	vec diag;
	
};

}
}
}



#endif
