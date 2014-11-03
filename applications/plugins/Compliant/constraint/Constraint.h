#ifndef COMPLIANT_CONSTRAINT_H
#define COMPLIANT_CONSTRAINT_H

#include "../initCompliant.h"
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa {
namespace component {
namespace linearsolver {

/// Base class to define the constraint type
struct SOFA_Compliant_API Constraint : public core::objectmodel::BaseObject {

    SOFA_ABSTRACT_CLASS(Constraint, sofa::core::objectmodel::BaseObject);
 
    virtual ~Constraint() {}
	
    /// project the response on the valid sub-space
    /// @correctionPass informs if the correction pass is performing (in which case only a friction projection should only treat the unilateral projection for example)
    virtual void project(SReal* out, unsigned n, unsigned index, bool correctionPass=false) const = 0;

    /// Flagging which constraints must be activated (true == active)
    /// ie filter out all deactivated constraints (force lambda to 0)
    /// If mask is empty, all constraints are activated
    /// A value per constraint block (NOT per constraint line)
    vector<bool> mask;
	
};

}
}
}


#endif // COMPLIANT_CONSTRAINT_H
