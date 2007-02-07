#ifndef SOFA_COMPONENT_COLLISION_NARROWPHASEDETECTION_H
#define SOFA_COMPONENT_COLLISION_NARROWPHASEDETECTION_H

#include <sofa/core/componentmodel/collision/Detection.h>
#include <vector>
#include <algorithm>

using namespace sofa::core::componentmodel::collision;

namespace sofa
{

namespace component
{

namespace collision
{

class NarrowPhaseDetection : virtual public Detection
{
protected:
    std::vector< std::pair<core::CollisionElementIterator, core::CollisionElementIterator> > elemPairs;

public:
    virtual ~NarrowPhaseDetection() { }

    virtual void addCollisionPair (const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair) = 0;

    virtual void addCollisionPairs(const std::vector< std::pair<core::CollisionModel*, core::CollisionModel*> > v)
    {
        for (std::vector< std::pair<core::CollisionModel*, core::CollisionModel*> >::const_iterator it = v.begin(); it<v.end(); it++)
            addCollisionPair(*it);
    }

    virtual void clearNarrowPhase()
    {
        elemPairs.clear();
    };

    std::vector<std::pair<core::CollisionElementIterator, core::CollisionElementIterator> >& getCollisionElementPairs() { return elemPairs; }
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
