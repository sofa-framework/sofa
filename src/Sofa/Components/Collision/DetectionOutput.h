#ifndef SOFA_COMPONENTS_COLLISION_DETECTIONOUTPUT_H
#define SOFA_COMPONENTS_COLLISION_DETECTIONOUTPUT_H

#include "Sofa/Abstract/CollisionElement.h"
#include "../Common/Vec.h"
#include <iostream>

namespace Sofa
{

namespace Components
{

namespace Collision
{

using namespace Common;

class DetectionOutput
{
public:
    std::pair<Abstract::CollisionElement*, Abstract::CollisionElement*> elem;
    Vector3 point[2];
    bool distance;
    double deltaT;
    DetectionOutput()
        : elem(NULL, NULL), distance(false), deltaT(0.0)
    {
    }
};

} // namespace Collision

} // namespace Components

} // namespace Sofa

#endif
