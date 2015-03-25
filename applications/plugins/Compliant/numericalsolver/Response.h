#ifndef COMPLIANT_RESPONSE_H
#define COMPLIANT_RESPONSE_H

#include "../initCompliant.h"

#include "../assembly/AssembledSystem.h"
#include "../utils/eigen_types.h"
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa {
namespace component {
namespace linearsolver {


/// Base class to solve the linear ode/dynamics equation
class Response : public core::objectmodel::BaseObject,
                 public utils::eigen_types {
public:
	SOFA_CLASS(Response, core::objectmodel::BaseObject);

	typedef AssembledSystem system_type;

    // FIXME remove mat altogether
    typedef rmat mat;

    
    /// @param semidefinite indicates if the matrix is semi-definite
    virtual void factor(const mat&, bool semidefinite=false ) = 0;
	
	virtual void solve(cmat&, const cmat& ) const = 0;
	virtual void solve(vec&, const vec& ) const = 0;

	
};


}
}
}

#endif
