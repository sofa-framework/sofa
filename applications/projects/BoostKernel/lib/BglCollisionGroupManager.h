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
#ifndef SOFA_COMPONENT_COLLISION_BGLCOLLISIONGROUPMANAGER_H
#define SOFA_COMPONENT_COLLISION_BGLCOLLISIONGROUPMANAGER_H

#include <sofa/core/componentmodel/collision/CollisionGroupManager.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/component/component.h>


namespace sofa
{

namespace component
{

namespace collision
{

class SOFA_COMPONENT_COLLISION_API BglCollisionGroupManager : public core::componentmodel::collision::CollisionGroupManager
{
public:
    typedef std::set<simulation::Node*> GroupSet;
    GroupSet groupSet;
public:
    BglCollisionGroupManager();

    virtual ~BglCollisionGroupManager();

    virtual void createGroups(core::objectmodel::BaseContext* scene, const sofa::helper::vector<core::componentmodel::collision::Contact*>& contacts);

    virtual void clearGroups(core::objectmodel::BaseContext* scene);

    /** Overload this if yo want to design your collision group, e.g. with a MasterSolver.
    Otherwise, an empty Node is returned.
    The OdeSolver is added afterwards.
    */
    virtual simulation::Node* buildCollisionGroup();

protected:
    template <typename ContainerParent>
    typename ContainerParent::value_type compatibleSetOfNode( ContainerParent &set1,ContainerParent &set2);
    virtual simulation::Node* getIntegrationNode(core::CollisionModel* model);

    std::map<Instance,GroupSet> storedGroupSet;

    virtual void changeInstance(Instance inst)
    {
        core::componentmodel::collision::CollisionGroupManager::changeInstance(inst);
        storedGroupSet[instance].swap(groupSet);
        groupSet.swap(storedGroupSet[inst]);
    }

};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
