#ifndef SOFA_CORE_COMPONENTMODEL_COLLISION_BROADPHASEDETECTION_H
#define SOFA_CORE_COMPONENTMODEL_COLLISION_BROADPHASEDETECTION_H

#include <sofa/core/componentmodel/collision/Detection.h>
#include <vector>
#include <algorithm>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace collision
{

class BroadPhaseDetection : virtual public Detection
{
protected:
    // it's an information to update the collisionMethod (like voxelgrid)
    int timeStamp;
    std::vector< std::pair<core::CollisionModel*, core::CollisionModel*> > cmPairs;

public:
    virtual ~BroadPhaseDetection() { }

    virtual void addCollisionModel(core::CollisionModel *cm) = 0;

    virtual void addCollisionModels(const std::vector<core::CollisionModel *> v)
    {
        for (std::vector<core::CollisionModel *>::const_iterator it = v.begin(); it<v.end(); it++)
            addCollisionModel(*it);
    }

    virtual void clearBroadPhase()
    {
        cmPairs.clear();
        cmNoCollision.clear();
    };

    std::vector<std::pair<core::CollisionModel*, core::CollisionModel*> >& getCollisionModelPairs() { return cmPairs; }
};

} // namespace collision

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
