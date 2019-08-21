#include "BulletCapsuleModel.inl"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace collision
{

int BulletCapsuleModelClass = core::RegisterObject("collision model using a set of Capsules, it can be used in the bullet collision pipeline")
        .add< TBulletCapsuleModel< defaulttype::Vec3Types> >()

        .addAlias("BulletCapsuleModel")
        .addAlias("BtCapsuleModel")
        .addAlias("BulletCapsule")
        .addAlias("BtCapsule")
        ;


int RigidBulletCapsuleModelClass = core::RegisterObject("collision model using a set of Capsules, it can be used in the bullet collision pipeline")
        .add< TBulletCapsuleModel< defaulttype::Rigid3Types> >()

        .addAlias("RigidBulletCapsuleModel")
        .addAlias("RigidBtCapsuleModel")
        .addAlias("RigidBulletCapsule")
        .addAlias("RigidBtCapsule")
        ;


template class SOFA_BULLETCOLLISIONDETECTION_API TBulletCapsuleModel<defaulttype::Vec3Types>;
template class SOFA_BULLETCOLLISIONDETECTION_API TBulletCapsuleModel<defaulttype::Rigid3Types>;


} // namespace collision

} // namespace component

} // namespace sofa
