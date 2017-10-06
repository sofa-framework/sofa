#include "MappingGeometricStiffnessForceField.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{
namespace constraint
{

int GeometricStiffnessForceFieldClass = sofa::core::RegisterObject("A ForceField that assembles the geometric stiffness stored in a Mapping")
#ifndef SOFA_FLOAT
.add<GeometricStiffnessForceField<sofa::defaulttype::Vec3dTypes> >()
.add<GeometricStiffnessForceField<sofa::defaulttype::Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
.add<GeometricStiffnessForceField<sofa::defaulttype::Vec3fTypes> >()
.add<GeometricStiffnessForceField<sofa::defaulttype::Rigid3fTypes> >();
#endif



}

}