#ifndef COMPLIANT_RESTITUTION_H
#define COMPLIANT_RESTITUTION_H

#include "ConstraintValue.h"

namespace sofa {
namespace component {
namespace odesolver {

// a constraint value for elastic contact (with restitution)
class SOFA_Compliant_API Restitution : public ConstraintValue {
  public:

    SOFA_CLASS(Restitution, ConstraintValue);

    Restitution() {}
    Restitution( mstate_type* mstate );

    // value for dynamics
    virtual void dynamics(SReal* dst, unsigned n) const;

    /// flagging which constraint lines must be activated
    // warning: the constraint can be created before intersection (alarm distance), in that case penetration depth is positive, and no constraint should be applied
    typedef vector<bool> mask_type;
    Data<mask_type> mask;

    Data<SReal> restitution;
};

}
}
}



#endif
