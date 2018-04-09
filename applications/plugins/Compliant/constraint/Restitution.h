#ifndef COMPLIANT_RESTITUTION_H
#define COMPLIANT_RESTITUTION_H

#include "ConstraintValue.h"
#include "Constraint.h"

namespace sofa {
namespace component {
namespace odesolver {

/// a constraint value for elastic contact (with restitution)
///
/// @author Matthieu Nesme
///
class SOFA_Compliant_API Restitution : public ConstraintValue
{
  public:

    SOFA_CLASS(Restitution, ConstraintValue);

    Restitution( mstate_type* mstate = 0 );

    // value for stabilization
    virtual void correction(SReal* dst, unsigned n, unsigned dim, const core::MultiVecCoordId& posId = core::VecCoordId::position(), const core::MultiVecDerivId& velId = core::VecDerivId::velocity()) const;
    // value for dynamics
    virtual void dynamics(SReal* dst, unsigned n, unsigned dim, bool, const core::MultiVecCoordId& posId = core::VecCoordId::position(), const core::MultiVecDerivId& velId = core::VecDerivId::velocity()) const;
    // flag violated constraints
    virtual void filterConstraints( helper::vector<bool>*& activateMask, const core::MultiVecCoordId& posId, unsigned n, unsigned dim );
    // clear violated mask
    virtual void clear() { mask.beginWriteOnly()->clear(); mask.endEdit(); }

    /// flagging which constraint blocks must be activated
    // warning: the constraint can be created before intersection (alarm distance), in that case penetration depth is positive, and no constraint should be applied
    typedef helper::vector<bool> mask_type;
    Data<mask_type> mask; ///< violated constraint

    Data<SReal> restitution; ///< restitution coefficient

};

}
}
}



#endif
