/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_DEFAULTCOLLISIONGROUPMANAGER_H
#define SOFA_COMPONENT_COLLISION_DEFAULTCOLLISIONGROUPMANAGER_H
#include "config.h"

#include <sofa/core/collision/CollisionGroupManager.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/DeleteVisitor.h>
#include <sofa/simulation/CleanupVisitor.h>


namespace sofa
{

namespace component
{

namespace collision
{

class SOFA_MISC_COLLISION_API DefaultCollisionGroupManager : public core::collision::CollisionGroupManager
{
public:
    SOFA_CLASS(DefaultCollisionGroupManager,sofa::core::collision::CollisionGroupManager);

    typedef std::set<simulation::Node::SPtr> GroupSet;
    GroupSet groupSet;

protected:
    DefaultCollisionGroupManager();

    virtual ~DefaultCollisionGroupManager();
	
private:
	DefaultCollisionGroupManager(const DefaultCollisionGroupManager& n) ;
	DefaultCollisionGroupManager& operator=(const DefaultCollisionGroupManager& n) ;

public:

    virtual void createGroups(core::objectmodel::BaseContext* scene, const sofa::helper::vector<core::collision::Contact::SPtr>& contacts);

    virtual void clearGroups(core::objectmodel::BaseContext* scene);

protected:

    //Find the node containing the ode solver used to animate the mechanical model associated to the collision model
    virtual simulation::Node* getIntegrationNode(core::CollisionModel* model);

    virtual void changeInstance(Instance inst)
    {
        core::collision::CollisionGroupManager::changeInstance(inst);
        storedGroupSet[instance].swap(groupSet);
        groupSet.swap(storedGroupSet[inst]);
    }

    template <typename Container>
    void clearGroup(const Container &inNodes, simulation::Node::SPtr group)
    {
        core::objectmodel::BaseNode::SPtr parent = *inNodes.begin();
        while(!group->child.empty()) parent->moveChild(*group->child.begin());

        simulation::CleanupVisitor cleanupvis(sofa::core::ExecParams::defaultInstance());
        cleanupvis.execute(group.get());
        simulation::DeleteVisitor vis(sofa::core::ExecParams::defaultInstance());
        vis.execute(group.get());
        group->detachFromGraph();
        //delete group;
        group.reset();
    }


    std::map<Instance,GroupSet> storedGroupSet;


};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
