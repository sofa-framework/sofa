#include "RigidCapsuleModel.inl"

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace helper;

SOFA_DECL_CLASS(RigidCapsule)

int RigidCapsuleModelClass = core::RegisterObject("Collision model which represents a set of rigid capsules")
#ifndef SOFA_FLOAT
        .add<  TCapsuleModel<defaulttype::Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add < TCapsuleModel<defaulttype::Rigid3fTypes> >()
#endif
        .addAlias("RigidCapsule")
        .addAlias("RigidCapsuleModel")
//.addAlias("CapsuleMesh")
//.addAlias("CapsuleSet")
        ;

#ifndef SOFA_FLOAT
template class SOFA_BASE_COLLISION_API TCapsule<defaulttype::Rigid3dTypes>;
template class SOFA_BASE_COLLISION_API TCapsuleModel<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BASE_COLLISION_API TCapsule<defaulttype::Rigid3fTypes>;
template class SOFA_BASE_COLLISION_API TCapsuleModel<defaulttype::Rigid3fTypes>;
#endif



}
}
}
