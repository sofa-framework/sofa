#ifndef COMPLIANT_DIAGONALRESPONSE_H
#define COMPLIANT_DIAGONALRESPONSE_H

#include <Compliant/config.h>
#include "Response.h"

namespace sofa {
namespace component {
namespace linearsolver {

/// Solving the dynamics equation when the matrix is diagonal
/// It is generally the case for a lumped mass (particles) with the stiffness in compliance and neglecting the geometric stiffness
/// @warning with "constant" set to true, mass and stiffness cannot be added dynamically (like a mouse-spring or penalities)
///
/// @author Matthieu Nesme
///
class SOFA_Compliant_API DiagonalResponse : public Response {
public:
	SOFA_CLASS(DiagonalResponse, Response);
	
	DiagonalResponse();

    void factor(const rmat& sys) override;
	void solve(cmat& lval, const cmat& rval) const override;
	virtual void solve(vec& lval,  const vec& rval) const;
    void reinit() override;

	const vec& diagonal() const { return diag; }

    /// reuse first inversion
    Data<bool> constant;
	
protected:

	vec diag;

    bool factorized;
	
};

}
}
}



#endif
