#ifndef SOFA_COMPONENT_COLLISION_DEFAULTCOLLISIONGROUPMANAGER_H
#define SOFA_COMPONENT_COLLISION_DEFAULTCOLLISIONGROUPMANAGER_H

#include <sofa/core/componentmodel/collision/CollisionGroupManager.h>
#include <sofa/simulation/tree/GNode.h>
#include <set>


namespace sofa
{

namespace component
{

namespace collision
{

class DefaultCollisionGroupManager : public core::componentmodel::collision::CollisionGroupManager
{
public:
    std::set<simulation::tree::GNode*> groupSet;
    std::vector<core::objectmodel::BaseContext*> groupVec;
public:
    DefaultCollisionGroupManager();

    virtual ~DefaultCollisionGroupManager();

    virtual void createGroups(core::objectmodel::BaseContext* scene, const std::vector<core::componentmodel::collision::Contact*>& contacts);

    virtual void clearGroups(core::objectmodel::BaseContext* scene);

    virtual const std::vector<core::objectmodel::BaseContext*>& getGroups() { return groupVec; }

protected:
    virtual simulation::tree::GNode* getIntegrationNode(core::CollisionModel* model);
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
