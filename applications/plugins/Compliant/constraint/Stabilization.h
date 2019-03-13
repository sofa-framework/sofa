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

    Stabilization( mstate_type* mstate = nullptr );

    /// flagging which constraint blocks must be stabilized (if empty, all constraints are stabilized)
    typedef helper::vector<bool> mask_type;
    Data<mask_type> mask; ///< dofs to be stabilized
	
	// value for stabilization
    void correction(SReal* dst, unsigned n, unsigned dim, const core::MultiVecCoordId& posId = core::VecCoordId::position(), const core::MultiVecDerivId& velId = core::VecDerivId::velocity()) const override;
	
	// value for dynamics
    void dynamics(SReal* dst, unsigned n, unsigned dim, bool stabilization, const core::MultiVecCoordId& posId = core::VecCoordId::position(), const core::MultiVecDerivId& velId = core::VecDerivId::velocity()) const override;
	
    // flag violated constraints
    virtual void filterConstraints( helper::vector<bool>*& activateMask, const core::MultiVecCoordId& posId, unsigned n, unsigned dim );

    // clear violated mask
    void clear() override { mask.beginWriteOnly()->clear(); mask.endEdit(); }
};

}
}
}



#endif
