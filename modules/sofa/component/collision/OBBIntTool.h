#ifndef OBBINTTOOL_H
#define OBBINTTOOL_H
#include <sofa/component/collision/OBBModel.h>
#include <sofa/component/collision/IntrOBBOBB.h>
#include <sofa/core/collision/DetectionOutput.h>
#include <sofa/component/collision/CapsuleModel.h>
#include <sofa/component/collision/IntrSphereOBB.h>
namespace sofa{
namespace component{
namespace collision{

class OBBIntTool{
public:
    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;
    typedef sofa::core::collision::DetectionOutput DetectionOutput;

    static bool computeIntersection(OBB&, OBB&,double alarmDist,double contactDist,OutputVector* contacts);
    static bool computeIntersection(Sphere &sphere, OBB &box,double alarmDist,double contactDist,OutputVector* contacts);
};

}
}
}
#endif // OBBINTTOOL_H
