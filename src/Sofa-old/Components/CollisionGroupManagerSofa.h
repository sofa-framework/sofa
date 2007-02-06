#ifndef SOFA_COMPONENTS_COLLISIONGROUPMANAGERSOFA_H
#define SOFA_COMPONENTS_COLLISIONGROUPMANAGERSOFA_H

#include "Collision/CollisionGroupManager.h"
#include "Graph/GNode.h"

#include <set>

namespace Sofa
{

namespace Components
{

class CollisionGroupManagerSofa : public Collision::CollisionGroupManager
{
public:
    std::set<Graph::GNode*> groupSet;
    std::vector<Abstract::BaseContext*> groupVec;
public:
    CollisionGroupManagerSofa();

    virtual ~CollisionGroupManagerSofa();

    virtual void createGroups(Abstract::BaseContext* scene, const std::vector<Collision::Contact*>& contacts);

    virtual void clearGroups(Abstract::BaseContext* scene);

    virtual const std::vector<Abstract::BaseContext*>& getGroups() { return groupVec; }

protected:
    virtual Graph::GNode* getIntegrationNode(Abstract::CollisionModel* model);
};

} // namespace Components

} // namespace Sofa

#endif
