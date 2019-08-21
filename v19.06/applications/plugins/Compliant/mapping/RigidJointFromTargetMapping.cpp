#include "RigidJointFromTargetMapping.h"

#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace defaulttype;

int RigidJointFromTargetMappingClass = core::RegisterObject("Computes relative rigid configurations from given targets")

.add< RigidJointFromTargetMapping< Rigid3Types, Vec6Types > >()

;

template class SOFA_Compliant_API RigidJointFromTargetMapping<  Rigid3Types, Vec6Types >;



///////////////////////


int RigidJointFromWorldFrameMappingClass = core::RegisterObject("Computes relative rigid configurations from world frame")

.add< RigidJointFromWorldFrameMapping< Rigid3Types, Vec6Types > >()

;

template class SOFA_Compliant_API RigidJointFromWorldFrameMapping<  Rigid3Types, Vec6Types >;






} // namespace mapping

} // namespace component

} // namespace sofa

