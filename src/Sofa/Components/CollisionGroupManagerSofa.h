#ifndef SOFA_COMPONENTS_COLLISIONGROUPMANAGERSOFA_H
#define SOFA_COMPONENTS_COLLISIONGROUPMANAGERSOFA_H

#include "Collision/CollisionGroupManager.h"
#include "Sofa/Core/MechanicalGroup.h"
#include "Scene.h"

#include <set>

namespace Sofa
{

namespace Components
{

class CollisionGroupManagerSofa : public Collision::CollisionGroupManager
{
public:
    std::string name;
    std::set<Core::MechanicalGroup*> groupSet;
    std::vector<Core::Group*> groupVec;
public:
    CollisionGroupManagerSofa(const std::string& name);

    virtual ~CollisionGroupManagerSofa();

    virtual const char* getName() { return name.c_str(); }

    virtual void createGroups(Scene* scene, const std::vector<Collision::Contact*>& contacts);

    virtual void clearGroups(Scene* scene);

    virtual const std::vector<Core::Group*>& getGroups() { return groupVec; }
};

} // namespace Components

} // namespace Sofa

#endif
