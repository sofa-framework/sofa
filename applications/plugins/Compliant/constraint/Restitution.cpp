#include "Restitution.h"

#include <sofa/core/ObjectFactory.h>
#include "../utils/map.h"

namespace sofa {
namespace component {
namespace odesolver {


SOFA_DECL_CLASS(Restitution);
int RestitutionClass = core::RegisterObject("Constraint value for elastic contact (with restitution)").add< Restitution >();


Restitution::Restitution( mstate_type* mstate )
    : ConstraintValue( mstate )
    , mask(initData(&mask, "mask", "violated constraint"))
    , restitution(initData(&restitution, SReal(0), "restitution", "restitution coefficient"))
{}


// WARNING can have interpenetration pbs if the violation veolcity is too small
// in that case we should switch to damped (or stabilized is no damping) constraints.


void Restitution::dynamics(SReal* dst, unsigned n) const
{
    // by default a regular position constraint (will be used for non-violated constraints, under alarm distance)
    ConstraintValue::dynamics(dst,n);

    const mask_type& mask = this->mask.getValue();

    // access contact velocity
    SReal* v = new SReal[n];
    mstate->copyToBuffer(v, core::VecDerivId::velocity(), n); // todo improve by accessing directly the i-th entrie without copy the entire array

    // create restitution for violated constraints (under contact distance)
    unsigned i = 0;
    for(SReal* last = dst + n; dst < last; ++dst, ++i) {
        if( mask[i] /*&& v[i] < epsilon*/ ) // TODO handle too small velocity
            *dst = -v[i] * restitution.getValue(); // we sneakily fake constraint error with reflected-adjusted relative velocity
    }

    delete [] v;

}






}
}
}
