#include "CylinderModel.inl"

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace helper;

SOFA_DECL_CLASS(Cylinder)

int RigidCylinderModelClass = core::RegisterObject("Collision model which represents a set of rigid cylinders")
#ifndef SOFA_FLOAT
        .add<  TCylinderModel<defaulttype::Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add < TCylinderModel<defaulttype::Rigid3fTypes> >()
#endif
        .addAlias("Cylinder")
        .addAlias("CylinderModel")
//.addAlias("CylinderMesh")
//.addAlias("CylinderSet")
        ;

#ifndef SOFA_FLOAT
template class SOFA_BASE_COLLISION_API TCylinder<defaulttype::Rigid3dTypes>;
template class SOFA_BASE_COLLISION_API TCylinderModel<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BASE_COLLISION_API TCylinder<defaulttype::Rigid3fTypes>;
template class SOFA_BASE_COLLISION_API TCylinderModel<defaulttype::Rigid3fTypes>;
#endif



}
}
}
