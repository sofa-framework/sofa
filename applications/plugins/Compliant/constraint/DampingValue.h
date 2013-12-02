#ifndef COMPLIANT_DAMPINGVALUE_H
#define COMPLIANT_DAMPINGVALUE_H

#include "ConstraintValue.h"

namespace sofa {
namespace component {
namespace odesolver {

// Constraint value for damping compliance
class SOFA_Compliant_API DampingValue : public ConstraintValue {
  public:

    SOFA_CLASS(DampingValue, ConstraintValue);

    DampingValue() {}
    DampingValue( mstate_type* mstate );

    // value for dynamics
    virtual void dynamics(SReal* dst, unsigned n) const;

};

}
}
}



#endif
