#ifndef COMPLIANT_CONSTRAINT_H
#define COMPLIANT_CONSTRAINT_H

#include "../initCompliant.h"
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa {
namespace component {
namespace linearsolver {

/// Base class to define the constraint type
struct SOFA_Compliant_API Constraint : core::objectmodel::BaseObject {

    SOFA_ABSTRACT_CLASS(Constraint, sofa::core::objectmodel::BaseObject);
 
    virtual ~Constraint() {};
	
    /// project the response on the valid sub-space
	virtual void project(SReal* out, unsigned n) const = 0;
	
};

}
}
}


#endif // COMPLIANT_CONSTRAINT_H
