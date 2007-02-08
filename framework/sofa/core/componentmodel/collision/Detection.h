#ifndef SOFA_CORE_COMPONENTMODEL_COLLISION_DETECTION_H
#define SOFA_CORE_COMPONENTMODEL_COLLISION_DETECTION_H

#include <sofa/core/CollisionModel.h>
#include <sofa/core/componentmodel/collision/Intersection.h>
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

class Detection : public virtual objectmodel::BaseObject
{
protected:
    /// Current intersection method
    Intersection* intersectionMethod;

    /// Contains the collisions models
    /// which are included in the broadphase
    /// but which are not in collisions with another model
    std::vector<core::CollisionModel*> cmNoCollision;
public:

    Detection()
        : intersectionMethod(NULL)
    {
    }

    /// virtual because subclasses might do precomputations based on intersection algorithms
    virtual void setIntersectionMethod(Intersection* v) { intersectionMethod = v;    }
    Intersection* getIntersectionMethod() const         { return intersectionMethod; }

    void removeCmNoCollision(core::CollisionModel* cm)
    {
        std::vector<core::CollisionModel*>::iterator it = std::find(cmNoCollision.begin(), cmNoCollision.end(), cm);
        if (it != cmNoCollision.end())
        {
            cmNoCollision.erase(it);
        }
    }

    void addNoCollisionDetect (core::CollisionModel* cm)
    {
        cmNoCollision.push_back(cm);
    }

    std::vector<core::CollisionModel*>& getListNoCollisionModel() {return cmNoCollision;};
};

} // namespace collision

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
