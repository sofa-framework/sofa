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


void Restitution::correction(SReal* dst, unsigned n, unsigned dim, const core::MultiVecCoordId& posId, const core::MultiVecDerivId& velId) const
{
    assert( mstate );

    unsigned size = n*dim;

    assert( mask.getValue().empty() || mask.getValue().size() == size );

    // copy constraint violation
    mstate->copyToBuffer(dst, posId.getId(mstate.get()), size);

    // access contact velocity
    SReal* v = new SReal[size];
    mstate->copyToBuffer(v, velId.getId(mstate.get()), size);

    const mask_type& mask = this->mask.getValue();

    unsigned i = 0;
    for(SReal* last = dst + size; dst < last; ++dst, ++i) {
        if( (!mask.empty() && !mask[i]) || std::fabs(v[i]) > HARD_CODED_VELOCITY_THRESHOLD )
            *dst = 0; // do not stabilize non violated or fast enough constraints
        else
            *dst = - *dst / this->getContext()->getDt(); // regular stabilized constraint
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

        if( ( mask.empty() || mask[line] ) ) // violated constraint
        {
            if( std::fabs(v[line]) > HARD_CODED_VELOCITY_THRESHOLD ) // the constraint is fast enough too be solved in pure velocity
            {
                dst[line] = - v[line] * restitution.getValue(); // we sneakily fake constraint error with reflected-adjusted relative velocity
            }
            else
            {
                if( stabilization ) dst[line] = 0;
                else dst[line] = - dst[line] / this->getContext()->getDt(); // regular non stabilized unilateral contact
            }

            //// next entries
            for( unsigned j=1 ; j<dim ; ++j )
                dst[line+j] = - dst[line+j] / this->getContext()->getDt(); // regular non stabilized unilateral contact

        }
        else // non-violated
        {
            // this value should have no influence
            // because a non-violated constraint with restitution should be deactivated
            // (ie flagged in Constraint::mask in projector)
            // and its lambda should be filtered out by the solver

            // to be more compatible with existing solver, let an approximation by enforcing to keep the same velocity
            for( unsigned j=0 ; j<dim ; ++j )
                dst[line+j] = v[line+j];
        }

    }

    delete [] v;
}


void Restitution::filterConstraints( std::vector<bool>& activateMask, const core::MultiVecCoordId& posId, unsigned n, unsigned dim )
{
    // non-violated constraints with restitution MUST be deactivated

    if( !mask.getValue().empty() ) return; // already be done (contact for example)

    unsigned size = n*dim;

    mask_type& mask = *this->mask.beginEdit();
    mask.resize( size );

    activateMask.resize( n );

    SReal* violation = new SReal[size];
    mstate->copyToBuffer(violation, posId.getId(mstate.get()), size);


    for( unsigned i=0 ; i<n ; ++i )
    {
        unsigned line = i*dim; // first contraint line
        if( violation[line]<0 ) // violated constraint
        {
            activateMask[i] = true;
            for( unsigned int j=0 ; j<dim ; ++j )
                mask[line+j]=1;
        }
        else
        {
            activateMask[i] = false;
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
