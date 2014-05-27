#include "UnilateralConstraint.h"
#include <sofa/core/ObjectFactory.h>

//#include <Eigen/Core>

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(UnilateralConstraint);
int UnilateralConstraintClass = core::RegisterObject("Unilateral constraint")
        .add< UnilateralConstraint >()
        .addAlias("UnilateralProjector"); // eheh :p


void UnilateralConstraint::project(SReal* out, unsigned n, bool) const
{
    for(unsigned i = 0; i < n; ++i)
        out[i] = std::max( (SReal)0.0, out[i] );
}


}
}
}
