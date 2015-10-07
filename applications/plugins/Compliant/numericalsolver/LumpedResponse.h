#ifndef COMPLIANT_LUMPEDRESPONSE_H
#define COMPLIANT_LUMPEDRESPONSE_H

#include <Compliant/config.h>
#include "Response.h"

namespace sofa {
namespace component {
namespace linearsolver {

/// Solving the dynamics equation after lumping the matrix (summing all line entries on the diagonal)
/// @warning its usage is really specific. Lumping the stiffness matrix does not make sense in the general case.
/// When the elasticity is in compliance, it can make sense to lump the geometric stiffness.
class SOFA_Compliant_API LumpedResponse : public Response {
public:
    SOFA_CLASS(LumpedResponse, Response);

    virtual void factor(const rmat& sys);
	virtual void solve(cmat& lval, const cmat& rval) const;
    virtual void solve(vec& lval,  const vec& rval) const;

	
protected:

	vec diag;
	
};

}
}
}



#endif
