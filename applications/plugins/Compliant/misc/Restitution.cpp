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

void Restitution::dynamics(SReal* dst, unsigned n) const
{
    assert( mstate );

    // we sneakily fake constraint error with reflected-adjusted relative velocity
    mstate->copyToBuffer(dst, core::VecDerivId::velocity(), n);
    map(dst, n) = -map(dst, n) * restitution.getValue();

    // TODO damping

    // zero for non violated
    const mask_type& mask = this->mask.getValue();
    unsigned i = 0;
    for(SReal* last = dst + n; dst < last; ++dst, ++i) {
        if( !mask[i] ) *dst = 0;
    }
}






}
}
}
