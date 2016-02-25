#include "RigidJointMultiMapping.h"

#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(RigidJointMultiMapping)

using namespace defaulttype;

// Register in the Factory
const int RigidJointMultiMappingClass = core::RegisterObject("Computes relative rigid configurations")

#ifndef SOFA_FLOAT
    .add< RigidJointMultiMapping< Rigid3dTypes, Vec6dTypes > >()
    .add< RigidJointMultiMapping< Rigid3dTypes, Rigid3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
    .add< RigidJointMultiMapping< Rigid3fTypes, Vec6fTypes > >()
    .add< RigidJointMultiMapping< Rigid3fTypes, Rigid3fTypes > >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API RigidJointMultiMapping<  Rigid3dTypes, Vec6dTypes >;
template class SOFA_Compliant_API RigidJointMultiMapping<  Rigid3dTypes, Rigid3dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API RigidJointMultiMapping< Rigid3fTypes, Vec6fTypes >;
template class SOFA_Compliant_API RigidJointMultiMapping< Rigid3fTypes, Rigid3fTypes >;
#endif



} // namespace mapping

} // namespace component

} // namespace sofa

