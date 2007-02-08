#ifndef SOFA_CORE_COMPONENTMODEL_COLLISION_COLLISIONGROUPMANAGER_H
#define SOFA_CORE_COMPONENTMODEL_COLLISION_COLLISIONGROUPMANAGER_H

#include <sofa/core/componentmodel/collision/Contact.h>
#include <sofa/core/objectmodel/BaseObject.h>

#include <vector>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace collision
{

//class Scene;

class CollisionGroupManager : public virtual objectmodel::BaseObject
{
public:
    virtual ~CollisionGroupManager() { }

    virtual void createGroups(objectmodel::BaseContext* scene, const std::vector<Contact*>& contacts) = 0;

    virtual void clearGroups(objectmodel::BaseContext* scene) = 0;

    virtual const std::vector<objectmodel::BaseContext*>& getGroups() = 0;
};

} // namespace collision

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
