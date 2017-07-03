#include "RigidOrientLogMapping.h"

#include <sofa/core/ObjectFactory.h>
#include <Compliant/mapping/CompliantMapping.inl>

namespace sofa {

namespace component {

namespace mapping {

SOFA_DECL_CLASS(RigidOrientLogMapping);

using namespace defaulttype;

static int handle  = core::RegisterObject("compute rigid orientation logarithm")

#ifndef SOFA_FLOAT
    .add< RigidOrientLogMapping< double > >()
#endif
#ifndef SOFA_DOUBLE
    .add< RigidOrientLogMapping< float > >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API RigidOrientLogMapping< double >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API RigidOrientLogMapping< float >;
#endif


}
}
}
   




