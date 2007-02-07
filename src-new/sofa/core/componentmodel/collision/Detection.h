#ifndef SOFA_COMPONENTS_COLLISION_DETECTION_H
#define SOFA_COMPONENTS_COLLISION_DETECTION_H

#include "Sofa-old/Abstract/CollisionModel.h"
#include "Intersection.h"
#include <vector>
#include <algorithm>

namespace Sofa
{

namespace Components
{

namespace Collision
{

class Detection : public virtual Abstract::BaseObject
{
protected:
    /// Current intersection method
    Intersection* intersectionMethod;

    /// Contains the collisions models
    /// which are included in the broadphase
    /// but which are not in collisions with another model
    std::vector<Abstract::CollisionModel*> cmNoCollision;
public:

    Detection()
        : intersectionMethod(NULL)
    {
    }

    /// virtual because subclasses might do precomputations based on intersection algorithms
    virtual void setIntersectionMethod(Intersection* v) { intersectionMethod = v;    }
    Intersection* getIntersectionMethod() const         { return intersectionMethod; }

    void removeCmNoCollision(Abstract::CollisionModel* cm)
    {
        std::vector<Abstract::CollisionModel*>::iterator it = std::find(cmNoCollision.begin(), cmNoCollision.end(), cm);
        if (it != cmNoCollision.end())
        {
            cmNoCollision.erase(it);
        }
    }

    void addNoCollisionDetect (Abstract::CollisionModel* cm)
    {
        cmNoCollision.push_back(cm);
    }

    std::vector<Abstract::CollisionModel*>& getListNoCollisionModel() {return cmNoCollision;};
};

} // namespace Collision

} // namespace Components

} // namespace Sofa

#endif
