#include "Restitution.h"

#include <sofa/core/ObjectFactory.h>

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


static const SReal HARD_CODED_VELOCITY_THRESHOLD = 1e-5; // TODO how to make it parameterizable?


void Restitution::correction(SReal* dst, unsigned n, const core::MultiVecCoordId& posId, const core::MultiVecDerivId& velId) const
{
    assert( mstate );
    assert( mask.getValue().empty() || mask.getValue().size() == n );

    // copy constraint violation
    mstate->copyToBuffer(dst, posId.getId(mstate.get()), n);

    // access contact velocity
    SReal* v = new SReal[n];
    mstate->copyToBuffer(v, velId.getId(mstate.get()), n);

    const mask_type& mask = this->mask.getValue();

    unsigned i = 0;
    for(SReal* last = dst + n; dst < last; ++dst, ++i) {
        if( (!mask.empty() && !mask[i]) || std::fabs(v[i]) > HARD_CODED_VELOCITY_THRESHOLD )
            *dst = 0; // do not stabilize non violated or fast enough constraints
        else
            *dst = - *dst / this->getContext()->getDt(); // regular stabilized constraint
    }

    delete [] v;
}



void Restitution::dynamics(SReal* dst, unsigned n, bool stabilization, const core::MultiVecCoordId& posId, const core::MultiVecDerivId& velId) const
{
    const mask_type& mask = this->mask.getValue();

    // copy constraint violation
    mstate->copyToBuffer(dst, posId.getId(mstate.get()), n);

    // access contact velocity
    SReal* v = new SReal[n];
    mstate->copyToBuffer(v, velId.getId(mstate.get()), n);


    // NOTE that if the violation velocity is too small this unstabilized velocity constraint can lead to interpenetration
    // in that case we switch to regular stabilized constraints.

    unsigned i = 0;
    for(SReal* last = dst + n; dst < last; ++dst, ++i)
    {
//        std::cerr<<"Restitution: "<<i<<" "<<v[i]<<std::endl;
        if( ( mask.empty() || mask[i] ) ) // violated constraint
        {
            if( std::fabs(v[i]) > HARD_CODED_VELOCITY_THRESHOLD ) // the constraint is fast enough too be solved in pure velocity
                *dst = - v[i] * restitution.getValue(); // we sneakily fake constraint error with reflected-adjusted relative velocity
            else
            {
                if( stabilization )
                    *dst = 0;
                else
                    *dst = - *dst / this->getContext()->getDt(); // regular non stabilized unilateral contact
            }
        }
        else // non-violated
        {
            *dst = v[i]; // enforce to keep the same velocity   TODO really do not consider the constraint with a special projector
        }
    }

    delete [] v;
}






}
}
}
