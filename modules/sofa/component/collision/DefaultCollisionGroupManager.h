/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_DEFAULTCOLLISIONGROUPMANAGER_H
#define SOFA_COMPONENT_COLLISION_DEFAULTCOLLISIONGROUPMANAGER_H

#include <sofa/core/collision/CollisionGroupManager.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/component/component.h>
#include <sofa/simulation/common/DeleteVisitor.h>


namespace sofa
{

namespace component
{

namespace collision
{

class SOFA_COMPONENT_COLLISION_API DefaultCollisionGroupManager : public core::collision::CollisionGroupManager
{
public:
    SOFA_CLASS(DefaultCollisionGroupManager,sofa::core::collision::CollisionGroupManager);

    typedef std::set<simulation::Node*> GroupSet;
    GroupSet groupSet;
public:
    DefaultCollisionGroupManager();

    virtual ~DefaultCollisionGroupManager();

    virtual void createGroups(core::objectmodel::BaseContext* scene, const sofa::helper::vector<core::collision::Contact*>& contacts);

    virtual void clearGroups(core::objectmodel::BaseContext* scene)=0;

protected:
    //Given to nodes, we find the parent node in common: if none is found, we return NULL
    virtual simulation::Node* findCommonParent(simulation::Node *group1, simulation::Node* group2)=0;

    //Find the node containing the ode solver used to animate the mechanical model associated to the collision model
    virtual simulation::Node* getIntegrationNode(core::CollisionModel* model);

    virtual void changeInstance(Instance inst)
    {
        core::collision::CollisionGroupManager::changeInstance(inst);
        storedGroupSet[instance].swap(groupSet);
        groupSet.swap(storedGroupSet[inst]);
    }

    template <typename Container, typename NodeType>
    void clearGroup(Container &inNodes, NodeType* group)
    {
        NodeType* parent = *inNodes.begin();
        while(!group->child.empty()) parent->moveChild(*group->child.begin());

        simulation::DeleteVisitor vis(sofa::core::ExecParams::defaultInstance());
        vis.execute(group);
        group->detachFromGraph();
        delete group;
    }


    std::map<Instance,GroupSet> storedGroupSet;


};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
