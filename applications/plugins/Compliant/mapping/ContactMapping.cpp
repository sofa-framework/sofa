#define SOFA_COMPONENT_MAPPING_ContactMapping_CPP

#include "ContactMapping.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace defaulttype;

// Register in the Factory
int ContactMappingClass = core::RegisterObject("Maps relative position/velocity between contact points")
.add< ContactMapping< Vec3Types, Vec1Types > >()
.add< ContactMapping< Vec3Types, Vec2Types > >()
.add< ContactMapping< Vec3Types, Vec3Types > >()

;

template class SOFA_Compliant_API ContactMapping<  Vec3Types, Vec1Types >;
template class SOFA_Compliant_API ContactMapping<  Vec3Types, Vec2Types >;
template class SOFA_Compliant_API ContactMapping<  Vec3Types, Vec3Types >;



} // namespace mapping

} // namespace component

} // namespace sofa


