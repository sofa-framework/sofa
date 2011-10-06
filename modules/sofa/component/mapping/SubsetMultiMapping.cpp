#define SOFA_COMPONENT_MAPPING_SUBSETMULTIMAPPING_CPP

#include <sofa/component/mapping/SubsetMultiMapping.inl>

#include <sofa/core/ObjectFactory.h>

using namespace sofa::defaulttype;

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(SubsetMultiMapping)

// Register in the Factory
int SubsetMultiMappingClass = core::RegisterObject("Compute a subset of the input MechanicalObjects according to a dof index list")
#ifndef SOFA_FLOAT
        .add< SubsetMultiMapping< Vec3dTypes, Vec3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< SubsetMultiMapping< Vec3fTypes, Vec3fTypes > >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_MISC_MAPPING_API SubsetMultiMapping< Vec3dTypes, Vec3dTypes >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_MISC_MAPPING_API SubsetMultiMapping< Vec3fTypes, Vec3fTypes >;
#endif

} // namespace mapping

} // namespace component

} // namespace sofa
