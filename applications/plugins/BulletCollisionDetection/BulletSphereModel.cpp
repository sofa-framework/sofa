#include "BulletSphereModel.inl"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace collision
{

int BulletSphereModelClass = core::RegisterObject("collision model using a set of spheres, it can be used in the bullet collision pipeline")
        .add< TBulletSphereModel<defaulttype::Vec3Types> >()

        .addAlias("BulletSphereModel")
        .addAlias("BtSphereModel")
        .addAlias("BulletSphere")
        .addAlias("BtSphere")
        ;

template class SOFA_BULLETCOLLISIONDETECTION_API TBulletSphereModel<defaulttype::Vec3Types>;


} // namespace collision

} // namespace component

} // namespace sofa
