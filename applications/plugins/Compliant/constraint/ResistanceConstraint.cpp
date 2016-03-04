#include "ResistanceConstraint.h"

#include <sofa/core/ObjectFactory.h>

#include <cmath>

namespace sofa {
namespace component {
namespace linearsolver {


SOFA_COMPLIANT_CONSTRAINT_CPP(ResistanceConstraint)

SOFA_DECL_CLASS(ResistanceConstraint)
int ResistanceConstraintClass = core::RegisterObject("Constraint creating a resistance (comparable to internal, dry friction in a joint)")
        .add< ResistanceConstraint >();



ResistanceConstraint::ResistanceConstraint()
    : Constraint()
    , d_threshold( initData(&d_threshold, "threshold", "The resistance force"))
{}

void ResistanceConstraint::project( SReal* out, unsigned n, unsigned /*index*/, bool correct ) const
{
    if( correct )
    {
        // no correction constraint
        memset( out, 0, n*sizeof(SReal) );
        return;
    }

    SReal t = d_threshold.getValue();
    for( unsigned int i=0 ; i<n ; ++i )
    {
        SReal o = std::abs( out[i] );
        if( o>t ) out[i] = helper::sign(out[i]) * t;
    }
}


}
}
}

