#define SOFA_COMPONENT_COLLISION_RIGIDCONTACTMAPPER_CPP
#include "BulletRigidContactMapper.inl"
#include <sofa/helper/Factory.inl>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace defaulttype;

SOFA_DECL_CLASS(BulletRigidContactMapper)

ContactMapperCreator< ContactMapper<BulletConvexHullModel,Vec3Types> > BulletConvexHullModelContactMapperClass("default", true);

template class SOFA_BULLETCOLLISIONDETECTION_API ContactMapper<BulletConvexHullModel,Vec3Types>;

} // namespace collision

} // namespace component

} // namespace sofa


