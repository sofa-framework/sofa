#define SOFA_COMPONENT_MAPPING_CONTACTMULTIMAPPING_CPP

#include "ContactMultiMapping.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(ContactMultiMapping)

using namespace defaulttype;

// Register in the Factory
int ContactMultiMappingClass = core::RegisterObject("Maps relative position/velocity between contact points")
#ifndef SOFA_FLOAT
.add< ContactMultiMapping< Vec3dTypes, Vec1dTypes > >()
.add< ContactMultiMapping< Vec3dTypes, Vec2dTypes > >()
.add< ContactMultiMapping< Vec3dTypes, Vec3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< ContactMultiMapping< Vec3fTypes, Vec1fTypes > >()
.add< ContactMultiMapping< Vec3fTypes, Vec2fTypes > >()
.add< ContactMultiMapping< Vec3fTypes, Vec3fTypes > >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API ContactMultiMapping<  Vec3dTypes, Vec1dTypes >;
template class SOFA_Compliant_API ContactMultiMapping<  Vec3dTypes, Vec2dTypes >;
template class SOFA_Compliant_API ContactMultiMapping<  Vec3dTypes, Vec3dTypes >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API ContactMultiMapping<  Vec3fTypes, Vec1fTypes >;
template class SOFA_Compliant_API ContactMultiMapping<  Vec3fTypes, Vec2fTypes >;
template class SOFA_Compliant_API ContactMultiMapping<  Vec3fTypes, Vec3fTypes >;
#endif


} // namespace mapping

} // namespace component

} // namespace sofa


