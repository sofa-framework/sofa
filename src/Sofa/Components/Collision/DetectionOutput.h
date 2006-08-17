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
    std::pair<Abstract::CollisionElementIterator, Abstract::CollisionElementIterator> elem; ///< Pair of colliding elements
    Vector3 point[2]; ///< Point in contact on each element
    Vector3 normal; ///< Normal of the contact, pointing outward from model 1
    //bool collision; ///< Are the elements interpenetrating
    double distance; ///< Distance between the elements (negative for interpenetration)
    double deltaT; ///< Time of contact (0 for non-continuous methods)
    DetectionOutput()
        : elem(NULL, NULL), distance(0.0), deltaT(0.0)
    {
    }
};

} // namespace Collision

} // namespace Components

} // namespace Sofa

#endif
