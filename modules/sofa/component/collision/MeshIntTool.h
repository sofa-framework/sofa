#ifndef MESHINTTOOL_H
#define MESHINTTOOL_H
#include <sofa/core/collision/Intersection.h>
#include <sofa/helper/FnDispatcher.h>
#include <sofa/component/collision/CapsuleModel.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/PointModel.h>
#include <sofa/component/collision/LineModel.h>

namespace sofa
{
namespace component
{
namespace collision
{


class MeshIntTool
{
public:
    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;

    static int computeIntersection(Capsule& cap, Point& pnt,double alarmDist,double contactDist,OutputVector* contacts);
    ////!\ CAUTION : uninitialized fields detection->elem and detection->id
    static int doCapPointInt(Capsule& cap, const Vector3& q,double alarmDist,double contactDist,OutputVector* contacts);

    static int computeIntersection(Capsule& cap, Line& lin,double alarmDist,double contactDist,OutputVector* contacts);

    ////!\ CAUTION : uninitialized fields detection->elem and detection->id
    static int doCapLineInt(Capsule& cap,const Vector3 & q1,const Vector3 & q2,double alarmDist,double contactDist,OutputVector* contacts,bool ignore_p1 = false,bool ignore_p2 = false);

    ////!\ CAUTION : uninitialized fields detection->elem and detection->id and detection->value
    static int doCapLineInt(const Vector3 & p1,const Vector3 & p2,double cap_rad,
                         const Vector3 & q1, const Vector3 & q2,double alarmDist,double contactDist,OutputVector* contacts,bool ignore_p1 = false,bool ignore_p2 = false);

    ////!\ CAUTION : uninitialized fields detection->elem and detection->id and detection->value, you have to substract contactDist, because
    ///this function can be used also as doIntersectionTriangleSphere where the contactDist = getContactDist() + sphere_radius
    static int doIntersectionTrianglePoint(double dist2, int flags, const Vector3& p1, const Vector3& p2, const Vector3& p3,const Vector3& q, OutputVector* contacts,bool swapElems = false);

    static int computeIntersection(Capsule& cap, Triangle& tri,double alarmDist,double contactDist,OutputVector* contacts);
};

}
}
}
#endif // CAPMESHINTTOOL_H
