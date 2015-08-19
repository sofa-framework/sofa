#include "Constraint.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa {
namespace component {
namespace linearsolver {

SOFA_DECL_CLASS(Constraint)
static int UnilateralConstraintClass = core::RegisterObject("basic constraint")
    .add< Constraint >();

Constraint::Constraint() : mask( NULL ) {}

void Constraint::project(SReal* /*out*/, unsigned /*n*/, unsigned /*index*/, bool /*correctionPass*/) const {
    // nothing lol
}




}
}
}
