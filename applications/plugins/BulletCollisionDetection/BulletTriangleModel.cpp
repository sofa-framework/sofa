#include "BulletTriangleModel.inl"

#include <sofa/core/ObjectFactory.h>
#include <SofaMeshCollision/BarycentricPenalityContact.h>
#include <sofa/core/collision/Contact.h>
namespace sofa
{

namespace component
{

namespace collision
{

int BulletTriangleModelClass = core::RegisterObject("collision model using a triangular mesh, as described in BaseMeshTopology, it can be used in the bullet collision pipeline")
#ifndef SOFA_FLOAT
        .add< TBulletTriangleModel<defaulttype::Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< TBulletTriangleModel<defaulttype::Vec3fTypes> >()
#endif
        .addAlias("BulletTriangleModel")
        .addAlias("BulletTriangleMeshModel")
        .addAlias("BulletTriangleSetModel")
        .addAlias("BulletTriangleMesh")
        .addAlias("BulletTriangleSet")
        .addAlias("BulletTriangle")
        ;


#ifndef SOFA_FLOAT
template class SOFA_BULLETCOLLISIONDETECTION_API TBulletTriangleModel<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BULLETCOLLISIONDETECTION_API TBulletTriangleModel<defaulttype::Vec3fTypes>;
#endif

} // namespace collision

} // namespace component

} // namespace sofa
