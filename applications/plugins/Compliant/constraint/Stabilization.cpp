#include "Stabilization.h"

#include <sofa/core/ObjectFactory.h>
#include "../utils/map.h"

namespace sofa {
namespace component {
namespace odesolver {


SOFA_DECL_CLASS(Stabilization)
int StabilizationClass = core::RegisterObject("Kinematic constraint stabilization").add< Stabilization >();


using namespace utils;

Stabilization::Stabilization( mstate_type* mstate )
    : BaseConstraintValue( mstate )
    , mask(initData(&mask, "mask", "dofs to be stabilized"))
{}

void Stabilization::correction(SReal* dst, unsigned n, unsigned dim, const core::MultiVecCoordId& posId, const core::MultiVecDerivId&) const {
	assert( mstate );

    unsigned size = n*dim;

    assert( mask.getValue().empty() || mask.getValue().size() == n );
	
    mstate->copyToBuffer(dst, posId.getId(mstate.get()), size);
    map(dst, size) *= -1.0 / this->getContext()->getDt();

	const mask_type& mask = this->mask.getValue();

    if( mask.empty() ) return; // all stabilized
	
    // zero for non-stabilized
	unsigned i = 0;
    for(SReal* last = dst + size; dst < last; dst+=dim, ++i) {
        if( !mask[i] ) memset( dst, 0, dim*sizeof(SReal) );
    }
}


void Stabilization::dynamics(SReal* dst, unsigned n, unsigned dim, bool stabilization, const core::MultiVecCoordId& posId, const core::MultiVecDerivId&) const {
    assert( mstate );

    const unsigned size = n*dim;

    // warning iff stabilization, only cancelling relative velocities of violated constraints (given by mask)


    assert( mask.getValue().empty() || mask.getValue().size() == n );
    const mask_type& mask = this->mask.getValue();

    if( !stabilization )
    {
        // elastic constraints
        mstate->copyToBuffer(dst, posId.getId(mstate.get()), size);
        map(dst, size) *= -1.0 / this->getContext()->getDt();
    }
    else
    {
        if( mask.empty() ) memset( dst, 0, size*sizeof(SReal) ); // all violated
        else
        {
            // for possible elastic constraint
            mstate->copyToBuffer(dst, posId.getId(mstate.get()), size);

            unsigned i = 0;
            for(SReal* last = dst + size; dst < last; dst+=dim, ++i) {
                if( mask[i] ) memset( dst, 0, dim*sizeof(SReal) ); // already violated
                else map(dst, dim) *= -1.0 / this->getContext()->getDt(); // not violated
            }
        }
    }
}



void Stabilization::filterConstraints( helper::vector<bool>* activateMask, const core::MultiVecCoordId& posId, unsigned n, unsigned dim )
{
    // All the constraints remain active
    // but non-violated constraint must not be stabilized

    if( !mask.getValue().empty() ) return; // already be done (contact for example)

    unsigned size = n*dim;

    mask_type& mask = *this->mask.beginWriteOnly();
    mask.resize( n );

    SReal* violation = new SReal[size];
    mstate->copyToBuffer(violation, posId.getId(mstate.get()), size);

    for( unsigned block=0 ; block<n ; ++block )
    {
        unsigned line = block*dim; // first constraint line
        if( violation[line]<0 ) // violated constraint
            mask[block]=true;
        else
            mask[block]=false;
    }

    this->mask.endEdit();

    delete [] violation;

    activateMask = &mask;
    (void) activateMask;
}


}
}
}
