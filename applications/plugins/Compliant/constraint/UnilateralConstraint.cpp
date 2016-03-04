#include "UnilateralConstraint.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa {
namespace component {
namespace linearsolver {


SOFA_COMPLIANT_CONSTRAINT_CPP(UnilateralConstraint)

SOFA_DECL_CLASS(UnilateralConstraint)
int UnilateralConstraintClass = core::RegisterObject("Unilateral constraint")
        .add< UnilateralConstraint >()
        .addAlias("UnilateralProjector"); // eheh :p


void UnilateralConstraint::project(SReal* out, unsigned n, unsigned, bool) const
{
    for(unsigned i = 0; i < n; ++i)
        if( out[i] < 0 ) out[i] = 0;
}


//////////////////////

SOFA_COMPLIANT_CONSTRAINT_CPP(NegativeUnilateralConstraint)

SOFA_DECL_CLASS(NegativeUnilateralConstraint)
int NegativeUnilateralConstraintClass = core::RegisterObject("Unilateral constraint")
        .add< NegativeUnilateralConstraint >();


void NegativeUnilateralConstraint::project(SReal* out, unsigned n, unsigned, bool) const
{
    for(unsigned i = 0; i < n; ++i)
        if( out[i] > 0 ) out[i] = 0;
}


}
}
}
