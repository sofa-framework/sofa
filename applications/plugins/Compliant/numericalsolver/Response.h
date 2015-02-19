#ifndef COMPLIANT_RESPONSE_H
#define COMPLIANT_RESPONSE_H

#include <Compliant/Compliant.h>

#include "../assembly/AssembledSystem.h"
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa {
namespace component {
namespace linearsolver {


/// Base class to solve the linear ode/dynamics equation
class Response : public core::objectmodel::BaseObject {
public:
	SOFA_CLASS(Response, core::objectmodel::BaseObject);

	typedef AssembledSystem system_type;
	
	typedef system_type::real real;
	typedef system_type::vec vec;

	typedef system_type::mat mat;
	typedef system_type::cmat cmat;


    /// @param semidefinite indicates if the matrix is semi-definite
    virtual void factor(const mat&, bool semidefinite=false ) = 0;
	
	virtual void solve(cmat&, const cmat& ) const = 0;
	virtual void solve(vec&, const vec& ) const = 0;

	
};


}
}
}

#endif
