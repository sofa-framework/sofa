#include "RigidLogMapping.h"

#include <sofa/core/ObjectFactory.h>
#include <Compliant/mapping/CompliantMapping.inl>

namespace sofa {

namespace component {

namespace mapping {

SOFA_DECL_CLASS(RigidLogMapping);

using namespace defaulttype;

static int handle  = core::RegisterObject("Computes rigid logarithm")

#ifndef SOFA_FLOAT
    .add< RigidLogMapping< double > >()
#endif
#ifndef SOFA_DOUBLE
    .add< RigidLogMapping< float > >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API RigidLogMapping< double >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API RigidLogMapping< float >;
#endif


}
}
}
   




