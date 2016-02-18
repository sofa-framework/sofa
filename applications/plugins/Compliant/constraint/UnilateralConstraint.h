#ifndef COMPLIANT_UNILATERALCONSTRAINT_H
#define COMPLIANT_UNILATERALCONSTRAINT_H

#include "Constraint.h"

namespace sofa {
namespace component {
namespace linearsolver {


/// Inequality constraint guaranteeing positiveness
struct SOFA_Compliant_API UnilateralConstraint : Constraint {
	
    SOFA_CLASS(UnilateralConstraint, Constraint);
    SOFA_COMPLIANT_CONSTRAINT_H( UnilateralConstraint )

    virtual void project(SReal* out, unsigned n, unsigned, bool=false) const;
	
};


/// Inequality constraint guaranteeing negativeness
struct SOFA_Compliant_API NegativeUnilateralConstraint : Constraint {

    SOFA_CLASS(NegativeUnilateralConstraint, Constraint);
    SOFA_COMPLIANT_CONSTRAINT_H( NegativeUnilateralConstraint )

    virtual void project(SReal* out, unsigned n, unsigned, bool=false) const;

};


}
}
}

#endif // COMPLIANT_UNILATERALCONSTRAINT_H
