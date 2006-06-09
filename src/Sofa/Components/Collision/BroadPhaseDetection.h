#ifndef SOFA_COMPONENTS_COLLISION_BROADPHASEDETECTION_H
#define SOFA_COMPONENTS_COLLISION_BROADPHASEDETECTION_H

#include "Detection.h"
#include <vector>
#include <algorithm>

namespace Sofa
{

namespace Components
{

namespace Collision
{

class BroadPhaseDetection : virtual public Detection
{
protected:
    // it's an information to update the collisionMethod (like voxelgrid)
    int timeStamp;
    std::vector< std::pair<Abstract::CollisionModel*, Abstract::CollisionModel*> > cmPairs;

public:
    virtual ~BroadPhaseDetection() { }

    virtual void addCollisionModel(Abstract::CollisionModel *cm) = 0;

    virtual void addCollisionModels(const std::vector<Abstract::CollisionModel *> v)
    {
        for (std::vector<Abstract::CollisionModel *>::const_iterator it = v.begin(); it<v.end(); it++)
            addCollisionModel(*it);
    }

    virtual void clearBroadPhase()
    {
        cmPairs.clear();
        cmNoCollision.clear();
    };

    std::vector<std::pair<Abstract::CollisionModel*, Abstract::CollisionModel*> >& getCollisionModelPairs() { return cmPairs; }
};

} // namespace Collision

} // namespace Components

} // namespace Sofa

#endif
