#include "RigidJointMultiMapping.h"

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
int RigidJointMultiMappingClass = core::RegisterObject("Computes relative rigid configurations")

    .add< RigidJointMultiMapping< Rigid3Types, Vec6Types > >()
    .add< RigidJointMultiMapping< Rigid3Types, Rigid3Types > >()

;

template class SOFA_Compliant_API RigidJointMultiMapping<  Rigid3Types, Vec6Types >;
template class SOFA_Compliant_API RigidJointMultiMapping<  Rigid3Types, Rigid3Types >;




} // namespace mapping

} // namespace component

} // namespace sofa

