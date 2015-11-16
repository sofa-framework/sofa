#include "RigidComMultiMapping.h"


#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(RigidComMultiMapping)

using namespace defaulttype;

// Register in the Factory
int RigidComMultiMappingClass = core::RegisterObject("Compute Center of Mass (CoM) for multi rigid dofs.")

#ifndef SOFA_FLOAT
.add< RigidComMultiMapping< Rigid3dTypes, Vec3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< RigidComMultiMapping< Rigid3fTypes, Vec3fTypes > >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API RigidComMultiMapping<  Rigid3dTypes, Vec3dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API RigidComMultiMapping< Rigid3fTypes, Vec3fTypes >;

#endif



} // namespace mapping

} // namespace component

} // namespace sofa

