#include "RigidJointMapping.h"

#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(RigidJointMapping)

using namespace defaulttype;

// Register in the Factory
int RigidJointMappingClass = core::RegisterObject("Computes relative rigid configurations")

#ifndef SOFA_FLOAT
.add< RigidJointMapping< Rigid3dTypes, Vec6dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< RigidJointMapping< Rigid3fTypes, Vec6fTypes > >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API RigidJointMapping<  Rigid3dTypes, Vec6dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API RigidJointMapping< Rigid3fTypes, Vec6fTypes >;
#endif



} // namespace mapping

} // namespace component

} // namespace sofa

