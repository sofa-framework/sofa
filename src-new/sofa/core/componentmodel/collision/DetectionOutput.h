#ifndef SOFA_CORE_COMPONENTMODEL_COLLISION_DETECTIONOUTPUT_H
#define SOFA_CORE_COMPONENTMODEL_COLLISION_DETECTIONOUTPUT_H

#include <sofa/core/CollisionElement.h>
#include <sofa/defaulttype/Vec.h>
#include <iostream>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace collision
{

using namespace sofa::defaulttype;

class DetectionOutput
{
public:
    std::pair<core::CollisionElementIterator, core::CollisionElementIterator> elem; ///< Pair of colliding elements
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

} // namespace collision

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
