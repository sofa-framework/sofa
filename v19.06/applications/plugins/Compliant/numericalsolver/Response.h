#ifndef COMPLIANT_RESPONSE_H
#define COMPLIANT_RESPONSE_H

#include <Compliant/config.h>

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
    SOFA_ABSTRACT_CLASS(Response, core::objectmodel::BaseObject);

	typedef AssembledSystem system_type;

    virtual void factor(const rmat&) = 0;
	
	virtual void solve(cmat&, const cmat& ) const = 0;
	virtual void solve(vec&, const vec& ) const = 0;

    // Does this factorization only work for symmetric matrices?
    virtual bool isSymmetric() const { return false; }
};


}
}
}

#endif
