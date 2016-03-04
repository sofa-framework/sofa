#include "RigidJointFromTargetMapping.h"

#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace defaulttype;

SOFA_DECL_CLASS(RigidJointFromTargetMapping)

int RigidJointFromTargetMappingClass = core::RegisterObject("Computes relative rigid configurations from given targets")

#ifndef SOFA_FLOAT
.add< RigidJointFromTargetMapping< Rigid3dTypes, Vec6dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< RigidJointFromTargetMapping< Rigid3fTypes, Vec6fTypes > >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API RigidJointFromTargetMapping<  Rigid3dTypes, Vec6dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API RigidJointFromTargetMapping< Rigid3fTypes, Vec6fTypes >;
#endif


///////////////////////


SOFA_DECL_CLASS(RigidJointFromWorldFrameMapping)

int RigidJointFromWorldFrameMappingClass = core::RegisterObject("Computes relative rigid configurations from world frame")

#ifndef SOFA_FLOAT
.add< RigidJointFromWorldFrameMapping< Rigid3dTypes, Vec6dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< RigidJointFromWorldFrameMapping< Rigid3fTypes, Vec6fTypes > >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API RigidJointFromWorldFrameMapping<  Rigid3dTypes, Vec6dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API RigidJointFromWorldFrameMapping< Rigid3fTypes, Vec6fTypes >;
#endif





} // namespace mapping

} // namespace component

} // namespace sofa

