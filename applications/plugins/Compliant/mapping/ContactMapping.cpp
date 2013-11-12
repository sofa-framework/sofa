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
int ContactMappingClass = core::RegisterObject("Maintains deltas at given distances")
#ifndef SOFA_FLOAT
.add< ContactMapping< Vec3dTypes, Vec1dTypes > >()
#endif

;

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API ContactMapping<  Vec3dTypes, Vec1dTypes >;
#endif



} // namespace mapping

} // namespace component

} // namespace sofa


