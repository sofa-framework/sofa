#define SOFA_COMPONENT_COLLISION_RIGIDCONTACTMAPPER_CPP
#include "BulletConvexHullContactMapper.h"
#include <sofa/helper/Factory.inl>
#include <SofaMeshCollision/RigidContactMapper.inl>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace defaulttype;

ContactMapperCreator< ContactMapper<BulletConvexHullModel,Vec3Types> > BulletConvexHullModelContactMapperClass("default", true);

template class SOFA_BULLETCOLLISIONDETECTION_API ContactMapper<BulletConvexHullModel,Vec3Types>;

} // namespace collision

} // namespace component

} // namespace sofa


