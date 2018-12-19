#define SOFA_COMPLIANT_RIGIDRESTJOINTMAPPING_CPP
#include "RigidRestJointMapping.h"

#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace defaulttype;

// Register in the Factory
int RigidRestJointMappingClass = core::RegisterObject("Computes relative rigid configurations from rest pos")

#ifndef SOFA_FLOAT
.add< RigidRestJointMapping< Rigid3dTypes, Vec6dTypes > >()
#endif
#ifndef SOFA_DOUBLE
.add< RigidRestJointMapping< Rigid3fTypes, Vec6fTypes > >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_Compliant_API RigidRestJointMapping<  Rigid3dTypes, Vec6dTypes >;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_Compliant_API RigidRestJointMapping< Rigid3fTypes, Vec6fTypes >;
#endif



} // namespace mapping

} // namespace component

} // namespace sofa

