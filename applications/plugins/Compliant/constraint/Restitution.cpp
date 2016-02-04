#include "Restitution.h"
#include "../utils/map.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa {
namespace component {
namespace odesolver {


SOFA_DECL_CLASS(Restitution)
int RestitutionClass = core::RegisterObject("Constraint value for elastic contact (with restitution)").add< Restitution >();

using namespace utils;

Restitution::Restitution( mstate_type* mstate )
    : ConstraintValue( mstate )
    , mask(initData(&mask, "mask", "violated constraint"))
    , restitution(initData(&restitution, SReal(0), "restitution", "restitution coefficient"))
{}


static const SReal HARD_CODED_VELOCITY_THRESHOLD = 1e-5; // TODO how to make it parameterizable?


void Restitution::correction(SReal* dst, unsigned n, unsigned dim, const core::MultiVecCoordId& posId, const core::MultiVecDerivId& velId) const
{
    assert( mstate );

    unsigned size = n*dim;

    assert( mask.getValue().empty() || mask.getValue().size() == n );

    // copy constraint violation
    mstate->copyToBuffer(dst, posId.getId(mstate.get()), size);

    // access contact velocity
    SReal* v = new SReal[size];
    mstate->copyToBuffer(v, velId.getId(mstate.get()), size);

    const mask_type& mask = this->mask.getValue();

    unsigned i = 0;
    for(SReal* last = dst + size; dst < last; dst+=dim, ++i) {
        if( (!mask.empty() && !mask[i]) || std::fabs(v[i*dim]) > HARD_CODED_VELOCITY_THRESHOLD )
            memset( dst, 0, dim*sizeof(SReal) ); // do not stabilize non violated or fast enough constraints
        else
            map(dst, dim) = -map(dst, dim) / this->getContext()->getDt(); // regular stabilized constraint
    }

    delete [] v;
}



void Restitution::dynamics(SReal* dst, unsigned n, unsigned dim, bool stabilization, const core::MultiVecCoordId& posId, const core::MultiVecDerivId& velId) const
{
    unsigned size = n*dim;

    const mask_type& mask = this->mask.getValue();

    // copy constraint violation
    mstate->copyToBuffer(dst, posId.getId(mstate.get()), size);

    // access contact velocity
    SReal* v = new SReal[size];
    mstate->copyToBuffer(v, velId.getId(mstate.get()), size);


    // NOTE that if the violation velocity is too small this unstabilized velocity constraint can lead to interpenetration
    // in that case we switch to regular stabilized constraints.



    for( unsigned i = 0 ; i<n ; ++i )
    {
        // WARNING constraint must be organized such as first value is along the normal

        //// first entry, along the normal
        unsigned line = i*dim;

        if( ( mask.empty() || mask[i] ) ) // violated constraint
        {
            if( std::fabs(v[line]) > HARD_CODED_VELOCITY_THRESHOLD ) // the constraint is fast enough too be solved in pure velocity
            {
                dst[line] = - v[line] * restitution.getValue(); // we sneakily fake constraint error with reflected-adjusted relative velocity

                //// next entries
                if( dim > 1 )
                {
                    if( stabilization ) memset( &dst[line+1], 0, (dim-1)*sizeof(SReal) ); // regular stabilized constraint
                    else map(&dst[line+1], dim-1) = -map(&dst[line+1], dim-1) / this->getContext()->getDt(); // regular elastic constraint
                }
            }
            else
            {
                if( stabilization ) memset( &dst[line], 0, dim*sizeof(SReal) ); // regular stabilized constraint
                else map(&dst[line], dim) = -map(&dst[line], dim) / this->getContext()->getDt(); // regular elastic constraint
            }

        }
        else // non-violated
        {
            // this value should have no influence
            // because a non-violated constraint with restitution should be deactivated
            // (ie flagged in Constraint::mask in projector)
            // and its lambda should be filtered out by the solver

            // to be more compatible with existing solvers, let an approximation by enforcing to keep the same velocity
            map(&dst[line], dim) = map(&v[line], dim);
        }

    }

    delete [] v;
}


void Restitution::filterConstraints( helper::vector<bool>* activateMask, const core::MultiVecCoordId& posId, unsigned n, unsigned dim )
{
    // non-violated constraints with restitution MUST be deactivated

    if( !mask.getValue().empty() ) return; // already be done (contact for example)

    unsigned size = n*dim;

    mask_type& mask = *this->mask.beginWriteOnly();
    mask.resize( n );


    SReal* violation = new SReal[size];
    mstate->copyToBuffer(violation, posId.getId(mstate.get()), size);


    for( unsigned block=0 ; block<n ; ++block )
    {
        unsigned line = block*dim; // first contraint line
        if( violation[line]<0 ) // violated constraint
        {
            mask[block]=true;
        }
        else
        {
            mask[block]=false;
        }

    }

    activateMask = &mask;
    (void) activateMask;

    this->mask.endEdit();
    delete [] violation;

}



}
}
}
