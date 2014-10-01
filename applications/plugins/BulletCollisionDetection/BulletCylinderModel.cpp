#include "BulletCylinderModel.inl"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace collision
{

int BulletCylinderModelClass = core::RegisterObject("collision model using a set of Capsules, it can be used in the bullet collision pipeline")
#ifndef SOFA_FLOAT
        .add< TBulletCylinderModel< defaulttype::Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< TBulletCylinderModel< defaulttype::Rigid3fTypes> >()
#endif
        .addAlias("BulletCylinderModel")
        .addAlias("BtCylinderModel")
        .addAlias("BulletCylinder")
        .addAlias("BtCylinder")
        ;


#ifndef SOFA_FLOAT
template class SOFA_BULLETCOLLISIONDETECTION_API TBulletCylinderModel<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BULLETCOLLISIONDETECTION_API TBulletCylinderModel<defaulttype::Rigid3fTypes>;
#endif

} // namespace collision

} // namespace component

} // namespace sofa
