#include "BulletCylinderModel.inl"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace collision
{

int BulletCylinderModelClass = core::RegisterObject("collision model using a set of Capsules, it can be used in the bullet collision pipeline")
        .add< TBulletCylinderModel< defaulttype::Rigid3Types> >()

        .addAlias("BulletCylinderModel")
        .addAlias("BtCylinderModel")
        .addAlias("BulletCylinder")
        .addAlias("BtCylinder")
        ;


template class SOFA_BULLETCOLLISIONDETECTION_API TBulletCylinderModel<defaulttype::Rigid3Types>;


} // namespace collision

} // namespace component

} // namespace sofa
