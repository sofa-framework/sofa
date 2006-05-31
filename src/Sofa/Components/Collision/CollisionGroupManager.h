#ifndef SOFA_COMPONENTS_COLLISION_COLLISIONGROUPMANAGER_H
#define SOFA_COMPONENTS_COLLISION_COLLISIONGROUPMANAGER_H

#include "Contact.h"
#include "Sofa/Abstract/BaseObject.h"

#include <vector>

namespace Sofa
{

namespace Components
{

class Scene;

namespace Collision
{

class CollisionGroupManager : public virtual Abstract::BaseObject
{
public:
    virtual ~CollisionGroupManager() { }

    virtual const char* getName() = 0;

    virtual void createGroups(Abstract::BaseContext* scene, const std::vector<Contact*>& contacts) = 0;

    virtual void clearGroups(Abstract::BaseContext* scene) = 0;

    virtual const std::vector<Abstract::BaseContext*>& getGroups() = 0;
};

} // namespace Collision

} // namespace Components

} // namespace Sofa

#endif
