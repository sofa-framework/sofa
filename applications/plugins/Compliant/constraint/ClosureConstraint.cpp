#include "ClosureConstraint.h"
#include <sofa/core/ObjectFactory.h>


namespace sofa {
namespace component {
namespace linearsolver {

SOFA_COMPLIANT_CONSTRAINT_CPP(ClosureConstraint)

int ClosureConstraintClass = core::RegisterObject("Closure constraint")
        .add< ClosureConstraint >();


}
}
}
