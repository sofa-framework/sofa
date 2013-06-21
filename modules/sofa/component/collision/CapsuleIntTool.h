#ifndef CAPSULEINTTOOL_H
#define CAPSULEINTTOOL_H
#include <sofa/core/collision/Intersection.h>
#include <sofa/helper/FnDispatcher.h>
#include <sofa/component/collision/CapsuleModel.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/collision/OBBModel.h>
#include <sofa/component/collision/IntrCapsuleOBB.h>
#include <cmath>

namespace sofa
{
namespace component
{
namespace collision
{
class SOFA_BASE_COLLISION_API CapsuleIntTool{
public:
    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;

    static int computeIntersection(Capsule&, Capsule&,double alarmDist,double contactDist,OutputVector* contacts);

    template <class DataTypes>
    static int computeIntersection(Capsule&, TSphere<DataTypes>&,double alarmDist,double contactDist,OutputVector* contacts);

    static int computeIntersection(Capsule&, OBB&,double alarmDist,double contactDist,OutputVector* contacts);
};

template <class DataTypes>
int CapsuleIntTool::computeIntersection(Capsule & cap, TSphere<DataTypes> & sph,double alarmDist,double contactDist,OutputVector* contacts){
    Vector3 sph_center = sph.center();
    Vector3 cap_p1 = cap.point1();
    Vector3 cap_p2 = cap.point2();
    SReal cap_rad = cap.radius();
    SReal sph_rad = sph.r();

    Vector3 AB = cap_p2 - cap_p1;
    Vector3 AC = sph_center - cap_p1;

    SReal theory_contactDist = (SReal) cap_rad + sph_rad + contactDist;
    SReal contact_exists = (SReal) cap_rad + sph_rad + alarmDist;
    SReal alpha = (SReal) (AB * AC)/AB.norm2();//projection of the sphere center on the capsule segment
                                        //alpha is the coefficient such as the projected point P = cap_p1 + alpha * AB
    if(alpha < 0.000001){//S is the sphere center, here is the case :
                         //        S
                         //           A--------------B
        Vector3 PQ = sph_center - cap_p1;

        if(PQ.norm2() > contact_exists * contact_exists)
            return 0;

        contacts->resize(contacts->size()+1);
        sofa::core::collision::DetectionOutput *detection = &*(contacts->end()-1);

        detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, sph);
        detection->id = (cap.getCollisionModel()->getSize() > sph.getCollisionModel()->getSize()) ? cap.getIndex() : sph.getIndex();

        detection->normal = PQ;
        detection->value = detection->normal.norm();
        detection->normal /= detection->value;
        detection->point[0] = cap_p1 + cap_rad * detection->normal;
        detection->point[1] = sph_center - sph_rad * detection->normal;
        detection->value -= theory_contactDist;

        return 1;
    }
    else if(alpha > 0.999999){//the case :
                              //                         S
                              //      A-------------B
        Vector3 PQ = sph_center - cap_p2;

        if(PQ.norm2() > contact_exists * contact_exists)
            return 0;

        contacts->resize(contacts->size()+1);
        sofa::core::collision::DetectionOutput *detection = &*(contacts->end()-1);

        detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, sph);
        detection->id = (cap.getCollisionModel()->getSize() > sph.getCollisionModel()->getSize()) ? cap.getIndex() : sph.getIndex();

        detection->normal = PQ;
        detection->value = detection->normal.norm();
        detection->normal /= detection->value;
        detection->point[0] = cap_p2 + cap_rad * detection->normal;
        detection->point[1] = sph_center - sph_rad * detection->normal;
        detection->value -= theory_contactDist;

        return 1;
    }
    else{//the case :
         //              S
         //      A-------------B
        Vector3 P = cap_p1 + alpha * AB;
        Vector3 PQ = sph_center - P;

        if(PQ.norm2() > contact_exists * contact_exists)
            return 0;

        contacts->resize(contacts->size()+1);
        sofa::core::collision::DetectionOutput *detection = &*(contacts->end()-1);

        detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(cap, sph);
        detection->id = (cap.getCollisionModel()->getSize() > sph.getCollisionModel()->getSize()) ? cap.getIndex() : sph.getIndex();

        detection->normal = PQ;
        detection->value = detection->normal.norm();
        detection->normal /= detection->value;
        detection->point[0] = P + cap_rad * detection->normal;
        detection->point[1] = sph_center - sph_rad * detection->normal;
        detection->value -= theory_contactDist;

        return 1;
    }
}

}
}
}
#endif // CAPSULEINTTOOL_H
