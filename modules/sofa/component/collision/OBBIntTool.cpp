#include <sofa/component/collision/OBBIntTool.h>

namespace sofa{
namespace component{
namespace collision{


bool OBBIntTool::computeIntersection(OBB & box0, OBB & box1,double alarmDist,double contactDist,OutputVector* contacts){
//    OBB::Real r02 = box0.extent(0)* box0.extent(0) + box0.extent(1)* box0.extent(1) + box0.extent(2)* box0.extent(2);
//    OBB::Real r12 = box1.extent(0)* box1.extent(0) + box1.extent(1)* box1.extent(1) + box1.extent(2)* box1.extent(2);
//    OBB::Real r0 = helper::rsqrt(r02);
//    OBB::Real r1 = helper::rsqrt(r12);
//    if((box0.center() - box1.center()).norm2() > r02 + 2*r0*r1 + r12){
//        return 0;
//    }
    IntrOBBOBB intr(box0,box1);
    //double max_time = helper::rsqrt((alarmDist * alarmDist)/((box1.lvelocity() - box0.lvelocity()).norm2()));
    if(/*intr.Find(max_time,box0.lvelocity(),box1.lvelocity())*/intr.FindStatic(alarmDist)){
        OBB::Real dist2 = (intr.GetPointOnFirst() - intr.GetPointOnSecond()).norm2();
        if((!intr.colliding()) && dist2 > alarmDist * alarmDist)
            return 0;

        contacts->resize(contacts->size()+1);
        DetectionOutput *detection = &*(contacts->end()-1);

        detection->normal = intr.separatingAxis();
        detection->point[0] = intr.GetPointOnFirst();
        detection->point[1] = intr.GetPointOnSecond();

        if(intr.colliding())
            detection->value = -helper::rsqrt(dist2) - contactDist;
        else
            detection->value = helper::rsqrt(dist2) - contactDist;

        detection->elem.first = box0;
        detection->elem.second = box1;
        detection->id = (box0.getCollisionModel()->getSize() > box1.getCollisionModel()->getSize()) ? box0.getIndex() : box1.getIndex();

        return 1;
    }
    else{
        std::cout<<"OUT"<<std::endl;
    }

    return 0;
}


}
}
}
