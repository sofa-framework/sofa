#ifndef SOFA_COMPONENTS_COLLISION_NARROWPHASEDETECTION_H
#define SOFA_COMPONENTS_COLLISION_NARROWPHASEDETECTION_H

#include "Detection.h"
#include <vector>
#include <algorithm>

namespace Sofa
{

namespace Components
{

namespace Collision
{

class NarrowPhaseDetection : virtual public Detection
{
protected:
    std::vector< std::pair<Abstract::CollisionElement*, Abstract::CollisionElement*> > elemPairs;

public:
    virtual ~NarrowPhaseDetection() { }

    virtual const char* getName() = 0;

    virtual void addCollisionPair (const std::pair<Abstract::CollisionModel*, Abstract::CollisionModel*>& cmPair) = 0;

    virtual void addCollisionPairs(const std::vector< std::pair<Abstract::CollisionModel*, Abstract::CollisionModel*> > v)
    {
        for (std::vector< std::pair<Abstract::CollisionModel*, Abstract::CollisionModel*> >::const_iterator it = v.begin(); it<v.end(); it++)
            addCollisionPair(*it);
    }

    virtual void clearNarrowPhase()
    {
        elemPairs.clear();
    };

    std::vector<std::pair<Abstract::CollisionElement*, Abstract::CollisionElement*> >& getCollisionElementPairs() { return elemPairs; }
};

} // namespace Collision

} // namespace Components

} // namespace Sofa

#endif
