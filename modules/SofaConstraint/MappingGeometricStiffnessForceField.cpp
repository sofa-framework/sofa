#include "MappingGeometricStiffnessForceField.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{
namespace constraint
{

int GeometricStiffnessForceFieldClass = sofa::core::RegisterObject("A ForceField that assembles the geometric stiffness stored in a Mapping")
.add<MappingGeometricStiffnessForceField<sofa::defaulttype::Vec3Types> >()
.add<MappingGeometricStiffnessForceField<sofa::defaulttype::Rigid3Types> >()

;


}

}
