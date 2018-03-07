#ifndef COMPLIANT_STABILIZATION_H
#define COMPLIANT_STABILIZATION_H

#include "BaseConstraintValue.h"

namespace sofa {
namespace component {
namespace odesolver {

/// W/o stabilization it results in Elastic constraints (cf ConstraintValue.h)
/// - the dynamics pass cancels the constraint violation
/// - no correction pass
///
/// W stabilisation it results in Holonomic Constraints
/// - the dynamics pass cancels relative velocity
/// - the correction pass cancels the constraint violation
class SOFA_Compliant_API Stabilization : public BaseConstraintValue {
  public:

    SOFA_CLASS(Stabilization, BaseConstraintValue);

    Stabilization( mstate_type* mstate = 0 );

    /// flagging which constraint blocks must be stabilized (if empty, all constraints are stabilized)
    typedef helper::vector<bool> mask_type;
    Data<mask_type> mask; ///< dofs to be stabilized
	
	// value for stabilization
    virtual void correction(SReal* dst, unsigned n, unsigned dim, const core::MultiVecCoordId& posId = core::VecCoordId::position(), const core::MultiVecDerivId& velId = core::VecDerivId::velocity()) const;
	
	// value for dynamics
    virtual void dynamics(SReal* dst, unsigned n, unsigned dim, bool stabilization, const core::MultiVecCoordId& posId = core::VecCoordId::position(), const core::MultiVecDerivId& velId = core::VecDerivId::velocity()) const;
	
    // flag violated constraints
    virtual void filterConstraints( helper::vector<bool>*& activateMask, const core::MultiVecCoordId& posId, unsigned n, unsigned dim );

    // clear violated mask
    virtual void clear() { mask.beginWriteOnly()->clear(); mask.endEdit(); }
};

}
}
}



#endif
