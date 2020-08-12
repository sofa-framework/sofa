#include "AssembledRigidRigidMapping.h"


#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace defaulttype;

// Register in the Factory
int AssembledRigidRigidMappingClass = core::RegisterObject("Assembled rigid displacement mapping")

.add< AssembledRigidRigidMapping< Rigid3Types, Rigid3Types > >()

;

template class SOFA_Compliant_API AssembledRigidRigidMapping<  Rigid3Types, Rigid3Types >;




} // namespace mapping

} // namespace component

} // namespace sofa

