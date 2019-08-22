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

.add< RigidRestJointMapping< Rigid3Types, Vec6Types > >()

;

template class SOFA_Compliant_API RigidRestJointMapping<  Rigid3Types, Vec6Types >;




} // namespace mapping

} // namespace component

} // namespace sofa

