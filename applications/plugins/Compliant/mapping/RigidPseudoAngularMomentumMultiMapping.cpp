#include "RigidPseudoAngularMomentumMultiMapping.h"


#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/core/ObjectFactory.h>


namespace sofa {
namespace component {
namespace mapping {

SOFA_DECL_CLASS(RigidPseudoAngularMomentumMultiMapping);

using namespace defaulttype;

// Register in the Factory
static const int handle = core::RegisterObject("compute advanced stuff")
    
#ifndef SOFA_FLOAT
.add< RigidPseudoAngularMomentumMultiMapping< Rigid3dTypes, Vec3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< RigidPseudoAngularMomentumMultiMapping< Rigid3fTypes, Vec3fTypes > >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API RigidPseudoAngularMomentumMultiMapping<  Rigid3dTypes, Vec3dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API RigidPseudoAngularMomentumMultiMapping< Rigid3fTypes, Vec3fTypes >;

#endif



} // namespace mapping
} // namespace component
} // namespace sofa

