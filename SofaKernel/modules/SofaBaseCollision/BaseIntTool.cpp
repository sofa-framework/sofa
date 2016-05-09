#include <SofaBaseCollision/BaseIntTool.h>

namespace sofa{namespace component{namespace collision{

//template<>
bool BaseIntTool::testIntersection(Cube &cube1, Cube &cube2,SReal alarmDist)
{
    const defaulttype::Vector3& minVect1 = cube1.minVect();
    const defaulttype::Vector3& minVect2 = cube2.minVect();
    const defaulttype::Vector3& maxVect1 = cube1.maxVect();
    const defaulttype::Vector3& maxVect2 = cube2.maxVect();

    for (int i = 0; i < 3; i++)
    {
        if ( minVect1[i] > maxVect2[i] + alarmDist || minVect2[i] > maxVect1[i] + alarmDist )
            return false;
    }

    return true;
}

class SOFA_BASE_COLLISION_API BaseIntTool;

}}}
