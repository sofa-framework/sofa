#include <sofa/component/collision/CapsuleIntTool.inl>

namespace sofa
{
namespace component
{
namespace collision
{
using namespace sofa::defaulttype;
using namespace sofa::core::collision;



template SOFA_BASE_COLLISION_API int CapsuleIntTool::computeIntersection(TCapsule<Vec3Types>&, TCapsule<Vec3Types>&,SReal alarmDist,SReal contactDist,OutputVector* contacts);
template SOFA_BASE_COLLISION_API int CapsuleIntTool::computeIntersection(TCapsule<Vec3Types>&, TCapsule<RigidTypes>&,SReal alarmDist,SReal contactDist,OutputVector* contacts);
template SOFA_BASE_COLLISION_API int CapsuleIntTool::computeIntersection(TCapsule<RigidTypes>&, TCapsule<RigidTypes>&,SReal alarmDist,SReal contactDist,OutputVector* contacts);
template SOFA_BASE_COLLISION_API int CapsuleIntTool::computeIntersection(TCapsule<RigidTypes> & cap, OBB& obb,SReal alarmDist,SReal contactDist,OutputVector* contacts);
template SOFA_BASE_COLLISION_API int CapsuleIntTool::computeIntersection(TCapsule<Vec3Types> & cap, OBB& obb,SReal alarmDist,SReal contactDist,OutputVector* contacts);

}
}
}
