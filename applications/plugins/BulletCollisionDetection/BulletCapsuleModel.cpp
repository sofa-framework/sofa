#include "BulletCapsuleModel.inl"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace collision
{

int BulletCapsuleModelClass = core::RegisterObject("collision model using a set of Capsules, it can be used in the bullet collision pipeline")
#ifndef SOFA_FLOAT
        .add< TBulletCapsuleModel< defaulttype::Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< TBulletCapsuleModel< defaulttype::Vec3fTypes> >()
#endif
        .addAlias("BulletCapsuleModel")
        .addAlias("BtCapsuleModel")
        .addAlias("BulletCapsule")
        .addAlias("BtCapsule")
        ;


int RigidBulletCapsuleModelClass = core::RegisterObject("collision model using a set of Capsules, it can be used in the bullet collision pipeline")
#ifndef SOFA_FLOAT
        .add< TBulletCapsuleModel< defaulttype::Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< TBulletCapsuleModel< defaulttype::Rigid3fTypes> >()
#endif
        .addAlias("RigidBulletCapsuleModel")
        .addAlias("RigidBtCapsuleModel")
        .addAlias("RigidBulletCapsule")
        .addAlias("RigidBtCapsule")
        ;


#ifndef SOFA_FLOAT
template class SOFA_BULLETCOLLISIONDETECTION_API TBulletCapsuleModel<defaulttype::Vec3dTypes>;
template class SOFA_BULLETCOLLISIONDETECTION_API TBulletCapsuleModel<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BULLETCOLLISIONDETECTION_API TBulletCapsuleModel<defaulttype::Vec3fTypes>;
template class SOFA_BULLETCOLLISIONDETECTION_API TBulletCapsuleModel<defaulttype::Rigid3fTypes>;
#endif

} // namespace collision

} // namespace component

} // namespace sofa
