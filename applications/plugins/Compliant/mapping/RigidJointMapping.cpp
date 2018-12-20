#include "RigidJointMapping.h"

#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace defaulttype;

// Register in the Factory
int RigidJointMappingClass = core::RegisterObject("Computes relative rigid configurations")

.add< RigidJointMapping< Rigid3Types, Vec6Types > >()

;

template class SOFA_Compliant_API RigidJointMapping<  Rigid3Types, Vec6Types >;




} // namespace mapping

} // namespace component

} // namespace sofa

