#ifndef SOFA_COMPONENTS_COLLISION_DETECTION_H
#define SOFA_COMPONENTS_COLLISION_DETECTION_H

#include "Sofa/Abstract/CollisionModel.h"
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
    /// Contains the collisions models
    /// which are included in the broadphase
    /// but which are not in collisions with another model
    std::vector<Abstract::CollisionModel*> cmNoCollision;
public:
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
