#include "RigidComMultiMapping.h"


#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace mapping
{

using namespace defaulttype;

// Register in the Factory
int RigidComMultiMappingClass = core::RegisterObject("Compute Center of Mass (CoM) for multi rigid dofs.")

.add< RigidComMultiMapping< Rigid3Types, Vec3Types > >()

;

template class SOFA_Compliant_API RigidComMultiMapping<  Rigid3Types, Vec3Types >;




} // namespace mapping

} // namespace component

} // namespace sofa

