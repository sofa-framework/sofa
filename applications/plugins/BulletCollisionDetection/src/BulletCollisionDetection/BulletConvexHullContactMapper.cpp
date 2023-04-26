#define SOFA_COMPONENT_COLLISION_RIGIDCONTACTMAPPER_CPP
#include "BulletConvexHullContactMapper.h"
#include <sofa/helper/Factory.inl>
#include <sofa/component/collision/response/mapper/RigidContactMapper.inl>


namespace sofa::component::collision::response::mapper
{

using namespace defaulttype;

ContactMapperCreator< ContactMapper<BulletConvexHullModel,Vec3Types> > BulletConvexHullModelContactMapperClass("PenalityContactForceField", true);

template class SOFA_BULLETCOLLISIONDETECTION_API ContactMapper<BulletConvexHullModel,Vec3Types>;

} // namespace sofa::component::collision::response::mapper
