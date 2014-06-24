#include <SofaBaseCollision/CapsuleIntTool.inl>

namespace sofa
{
namespace component
{
namespace collision
{
using namespace sofa::defaulttype;
using namespace sofa::core::collision;

bool CapsuleIntTool::shareSameVertex(const Capsule & c1,const Capsule & c2){
    return c1.shareSameVertex(c2);
}

template SOFA_BASE_COLLISION_API int CapsuleIntTool::computeIntersection(TCapsule<Vec3Types>&, TCapsule<Vec3Types>&,SReal alarmDist,SReal contactDist,OutputVector* contacts);
template SOFA_BASE_COLLISION_API int CapsuleIntTool::computeIntersection(TCapsule<Vec3Types>&, TCapsule<RigidTypes>&,SReal alarmDist,SReal contactDist,OutputVector* contacts);
template SOFA_BASE_COLLISION_API int CapsuleIntTool::computeIntersection(TCapsule<RigidTypes>&, TCapsule<RigidTypes>&,SReal alarmDist,SReal contactDist,OutputVector* contacts);
template SOFA_BASE_COLLISION_API int CapsuleIntTool::computeIntersection(TCapsule<RigidTypes> & cap, OBB& obb,SReal alarmDist,SReal contactDist,OutputVector* contacts);
template SOFA_BASE_COLLISION_API int CapsuleIntTool::computeIntersection(TCapsule<Vec3Types> & cap, OBB& obb,SReal alarmDist,SReal contactDist,OutputVector* contacts);

class SOFA_BASE_VISUAL_API CapsuleIntTool;
}
}
}
