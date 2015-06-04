#ifndef COMPLIANT_UNILATERALCONSTRAINT_H
#define COMPLIANT_UNILATERALCONSTRAINT_H

#include "Constraint.h"

namespace sofa {
namespace component {
namespace linearsolver {

struct SOFA_Compliant_API UnilateralConstraint : Constraint {
	
    SOFA_CLASS(UnilateralConstraint, Constraint);

    // WARNING: index is not used (see Constraint.h)
    virtual void project(SReal* out, unsigned n, unsigned /*index*/,
                         bool correctionPass=false) const;
	
};


}
}
}

#endif // COMPLIANT_UNILATERALCONSTRAINT_H
