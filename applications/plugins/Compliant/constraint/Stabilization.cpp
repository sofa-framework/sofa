#include "Stabilization.h"

#include <sofa/core/ObjectFactory.h>
#include "../utils/map.h"

namespace sofa {
namespace component {
namespace odesolver {


SOFA_DECL_CLASS(Stabilization);
int StabilizationClass = core::RegisterObject("Kinematic constraint stabilization").add< Stabilization >();


using namespace utils;

Stabilization::Stabilization( mstate_type* mstate )
    : BaseConstraintValue( mstate )
    , mask(initData(&mask, "mask", "dofs to be stabilized"))
    , m_holonomic( false )
{}

void Stabilization::correction(SReal* dst, unsigned n, unsigned dim, const core::MultiVecCoordId& posId, const core::MultiVecDerivId&) const {
	assert( mstate );

    unsigned size = n*dim;

    assert( mask.getValue().empty() || mask.getValue().size() == size );
	
    mstate->copyToBuffer(dst, posId.getId(mstate.get()), size);
	
	// TODO needed ?
    map(dst, size) = -map(dst, size) / this->getContext()->getDt();

	const mask_type& mask = this->mask.getValue();
	
	// non-zero for stabilized
	unsigned i = 0;
    for(SReal* last = dst + size; dst < last; ++dst, ++i) {
		if( !mask.empty() && !mask[i] ) *dst = 0;
    }
}


void Stabilization::dynamics(SReal* dst, unsigned n, unsigned dim, bool stabilization, const core::MultiVecCoordId& posId, const core::MultiVecDerivId&) const {
	assert( mstate );

    unsigned size = n*dim;

    assert( mask.getValue().empty() || mask.getValue().size() == size );

    mstate->copyToBuffer(dst, posId.getId(mstate.get()), size);
    map(dst, size) = -map(dst, size) / this->getContext()->getDt();

    // if there is no stabilization, the constraint must be corrected by the dynamics pass
    // if there is stabilization, the velocities will be corrected by the correction pass
    if( stabilization || m_holonomic )
    {
        const mask_type& mask = this->mask.getValue();
        // zero for stabilized, since the position error will be handled by the correction
        unsigned i = 0;
        for(SReal* last = dst + size; dst < last; ++dst, ++i) {
            if( mask.empty() || mask[i] ) *dst = 0;
        }
    }
}



void Stabilization::filterConstraints( std::vector<bool>& activateMask, const core::MultiVecCoordId& posId, unsigned n, unsigned dim )
{
    // All the constraints remain active
    // but non-violated constraint must not be stabilized

    if( !mask.getValue().empty() ) return; // already be done (contact for example)

    unsigned size = n*dim;

    mask_type& mask = *this->mask.beginEdit();
    mask.resize( size );

    activateMask.clear(); // all activated

    SReal* violation = new SReal[size];
    mstate->copyToBuffer(violation, posId.getId(mstate.get()), size);

    for( unsigned i=0 ; i<n ; ++i )
    {
        unsigned line = i*dim; // first constraint line
        if( violation[line]<0 ) // violated constraint
        {
            mask[line]=1;
            for( unsigned int j=1 ; j<dim ; ++j )
                mask[line+j]=0;
        }
        else
        {
            for( unsigned int j=0 ; j<dim ; ++j )
                mask[line+j]=0;
        }
    }

    this->mask.endEdit();
    delete [] violation;
}


}
}
}
