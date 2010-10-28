#define SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTI2MAPPING_CPP

#include <sofa/component/mapping/CenterOfMassMulti2Mapping.inl>

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(CenterOfMassMulti2Mapping)

using namespace sofa::defaulttype;

// Register in the Factory
int CenterOfMassMulti2MappingClass = core::RegisterObject("Set the point to the center of mass of the DOFs it is attached to")
#ifndef SOFA_FLOAT
        .add< CenterOfMassMulti2Mapping< Vec3dTypes, Rigid3dTypes, Vec3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< CenterOfMassMulti2Mapping< Vec3fTypes, Rigid3fTypes, Vec3fTypes > >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_MAPPING_API CenterOfMassMulti2Mapping< Vec3dTypes, Rigid3dTypes, Vec3dTypes >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_MAPPING_API CenterOfMassMulti2Mapping< Vec3fTypes, Rigid3fTypes, Vec3fTypes >;
#endif

} // namespace mapping

} // namespace component

} // namespace sofa
