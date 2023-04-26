#include "BulletTriangleModel.inl"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/Contact.h>
namespace sofa
{

namespace component
{

namespace collision
{

int BulletTriangleModelClass = core::RegisterObject("collision model using a triangular mesh, as described in BaseMeshTopology, it can be used in the bullet collision pipeline")
        .add< TBulletTriangleModel<defaulttype::Vec3Types> >()

        .addAlias("BulletTriangleModel")
        .addAlias("BulletTriangleMeshModel")
        .addAlias("BulletTriangleSetModel")
        .addAlias("BulletTriangleMesh")
        .addAlias("BulletTriangleSet")
        .addAlias("BulletTriangle")
        ;


template class SOFA_BULLETCOLLISIONDETECTION_API TBulletTriangleModel<defaulttype::Vec3Types>;


} // namespace collision

} // namespace component

} // namespace sofa
