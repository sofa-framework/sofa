#define SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTIMAPPING_CPP

#include <sofa/component/mapping/CenterOfMassMultiMapping.inl>

#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(CenterOfMassMultiMapping)

using namespace sofa::defaulttype;


// Register in the Factory
int CenterOfMassMultiMappingClass = core::RegisterObject("Set the point to the center of mass of the DOFs it is attached to")
#ifndef SOFA_FLOAT
        .add< CenterOfMassMultiMapping< Vec3dTypes, Vec3dTypes > >()
        .add< CenterOfMassMultiMapping< Rigid3dTypes, Rigid3dTypes > >()
        .add< CenterOfMassMultiMapping< Rigid3dTypes, Vec3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< CenterOfMassMultiMapping< Vec3fTypes, Vec3fTypes > >()
        .add< CenterOfMassMultiMapping< Rigid3fTypes, Rigid3fTypes > >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_MISC_MAPPING_API CenterOfMassMultiMapping< Vec3dTypes, Vec3dTypes >;
template class SOFA_MISC_MAPPING_API CenterOfMassMultiMapping< Rigid3dTypes, Rigid3dTypes >;
template class SOFA_MISC_MAPPING_API CenterOfMassMultiMapping< Rigid3dTypes, Vec3dTypes >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_MISC_MAPPING_API CenterOfMassMultiMapping< Vec3fTypes, Vec3fTypes >;
template class SOFA_MISC_MAPPING_API CenterOfMassMultiMapping< Rigid3fTypes, Rigid3fTypes >;
#endif

} // namespace mapping

} // namespace component

} // namespace sofa
