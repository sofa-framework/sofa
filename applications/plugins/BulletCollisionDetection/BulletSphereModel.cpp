#include "BulletSphereModel.inl"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace collision
{

int BulletSphereModelClass = core::RegisterObject("collision model using a set of spheres, it can be used in the bullet collision pipeline")
#ifndef SOFA_FLOAT
        .add< TBulletSphereModel<defaulttype::Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< TBulletSphereModel<defaulttype::Vec3fTypes> >()
#endif
        .addAlias("BulletSphereModel")
        .addAlias("BtSphereModel")
        .addAlias("BulletSphere")
        .addAlias("BtSphere")
        ;

#ifndef SOFA_FLOAT
template class SOFA_BULLETCOLLISIONDETECTION_API TBulletSphereModel<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BULLETCOLLISIONDETECTION_API TBulletSphereModel<defaulttype::Vec3fTypes>;
#endif

} // namespace collision

} // namespace component

} // namespace sofa
