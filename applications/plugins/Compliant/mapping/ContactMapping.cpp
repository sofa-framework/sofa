#define SOFA_COMPONENT_MAPPING_ContactMapping_CPP

#include "ContactMapping.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(ContactMapping)

using namespace defaulttype;

// Register in the Factory
int ContactMappingClass = core::RegisterObject("Maps relative position/velocity between contact points")
#ifndef SOFA_FLOAT
.add< ContactMapping< Vec3dTypes, Vec1dTypes > >()
.add< ContactMapping< Vec3dTypes, Vec3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< ContactMapping< Vec3fTypes, Vec1fTypes > >()
.add< ContactMapping< Vec3fTypes, Vec3fTypes > >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API ContactMapping<  Vec3dTypes, Vec1dTypes >;
template class SOFA_Compliant_API ContactMapping<  Vec3dTypes, Vec3dTypes >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API ContactMapping<  Vec3fTypes, Vec1fTypes >;
template class SOFA_Compliant_API ContactMapping<  Vec3fTypes, Vec3fTypes >;
#endif


} // namespace mapping

} // namespace component

} // namespace sofa


