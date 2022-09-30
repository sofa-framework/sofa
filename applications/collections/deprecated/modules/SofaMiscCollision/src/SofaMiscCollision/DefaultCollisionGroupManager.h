/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <SofaMiscCollision/config.h>

#include <sofa/core/collision/CollisionGroupManager.h>
#include <sofa/simulation/DeleteVisitor.h>
#include <sofa/simulation/CleanupVisitor.h>
#include <sofa/simulation/Node.h>
#include <SofaMiscCollision/SolverMerger.h>

namespace sofa::component::collision
{

class SOFA_MISC_COLLISION_API DefaultCollisionGroupManager : public core::collision::CollisionGroupManager
{
public:
    SOFA_CLASS(DefaultCollisionGroupManager,sofa::core::collision::CollisionGroupManager);



public:
    void createGroups(core::objectmodel::BaseContext* scene, const sofa::type::vector<core::collision::Contact::SPtr>& contacts) override;
    void clearGroups(core::objectmodel::BaseContext* scene) override;

protected:
    DefaultCollisionGroupManager();
    ~DefaultCollisionGroupManager() override;

    using GroupMap = std::map<simulation::Node*, simulation::Node*>;

    /// Keys are Nodes associated to a collision model which have been reorganized so two collision models in contact
    /// are under the same Node with a single solver.
    /// Values are Nodes in which the Node in key has been merged
    GroupMap groupMap;

    /// Map used in case a Node has been merged into another Node
    using MergeGroupsMap = std::map<simulation::Node*, simulation::Node*>;

    //Find the node containing the ode solver used to animate the mechanical model associated to the collision model
    virtual simulation::Node* getIntegrationNode(core::CollisionModel* model);

    void changeInstance(Instance inst) override;

    static void clearCollisionGroup(simulation::NodeSPtr group);

    std::map<Instance,GroupMap> storedGroupSet;

    void createGroup(core::collision::Contact* contact,
                     int& groupIndex,
                     MergeGroupsMap& mergedGroups,
                     sofa::type::vector< std::pair<core::collision::Contact*, simulation::Node::SPtr> >& contactGroup,
                     sofa::type::vector< simulation::Node::SPtr >& removedGroup,
                     sofa::type::vector< core::collision::Contact* >& stationaryContacts);

    /// Move all objects from one node to another
    static void moveAllObjects(simulation::Node::SPtr sourceNode, simulation::Node::SPtr destinationNode);

    /// Move all children from one node to another
    static void moveAllChildren(simulation::Node::SPtr sourceNode, simulation::Node::SPtr destinationNode);

    /// Get the ODE solver, the linear solver and the constraint solver in a node
    static void getSolverSet(simulation::Node::SPtr node, sofa::component::collision::SolverSet& outSolverSet);

    /// Remove the ODE solver, the linear solver and the constraint solver in a node
    static void removeSolverSetFromNode(simulation::Node::SPtr node, sofa::component::collision::SolverSet& solverSet);

    /// Destroy the instances of a set containing an ODE solver, a linear solver and a constraint solver
    static void destroySolvers(sofa::component::collision::SolverSet& solverSet);

    /// Add the ODE solver, the linear solver and the constraint solver into a node
    /// Also initializes the solvers
    static void addSolversToNode(simulation::Node::SPtr node, sofa::component::collision::SolverSet& solverSet);

    /// Given a Node node, search if it has been merged into another Node. Returns the Node in which node has been merged
    simulation::Node* getNodeFromMergedGroups(simulation::Node* node, const MergeGroupsMap& mergedGroups);

private:
    DefaultCollisionGroupManager(const DefaultCollisionGroupManager& n) ;
    DefaultCollisionGroupManager& operator=(const DefaultCollisionGroupManager& n) ;

    /// In the loop where it is called, this function checks if currentGroup == firstGroup, i.e. when the loop is endless, and throw and exception
    void checkEndlessLoop(const MergeGroupsMap& mergedGroups, simulation::Node* firstGroup, simulation::Node* currentGroup);
};

} // namespace sofa::component::collision