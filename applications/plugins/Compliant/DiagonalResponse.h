#ifndef COMPLIANT_DIAGONALRESPONSE_H
#define COMPLIANT_DIAGONALRESPONSE_H

#include "Response.h"

namespace sofa {
namespace component {
namespace linearsolver {

class DiagonalResponse : public Response {
public:
	SOFA_CLASS(DiagonalResponse, Response);
	
	virtual void factor(const mat& sys);
	virtual void solve(cmat& lval, const cmat& rval) const;
	virtual void solve(vec& lval,  const vec& rval) const;
	
protected:

	vec diagonal;
	
};

}
}
}



#endif
