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
#include <SofaMiscCollision/DefaultCollisionGroupManager.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

namespace sofa::component::collision
{

using core::collision::Contact;
using core::objectmodel::BaseNode;


int DefaultCollisionGroupManagerClass = core::RegisterObject("Responsible for gathering colliding objects in the same group, for consistent time integration")
        .add< DefaultCollisionGroupManager >()
        .addAlias( "CollisionGroupManager" )
        .addAlias( "CollisionGroup" )
        .addAlias( "TreeCollisionGroupManager" ) // for backward compatibility with old scene files but could be removed
        ;

DefaultCollisionGroupManager::DefaultCollisionGroupManager() = default;
DefaultCollisionGroupManager::~DefaultCollisionGroupManager() = default;

void DefaultCollisionGroupManager::clearCollisionGroup(simulation::Node::SPtr group)
{
    const simulation::Node::Parents &parents = group->getParents();
    core::objectmodel::BaseNode::SPtr parent = *parents.begin();
    while (!group->child.empty()) 
        parent->moveChild(core::objectmodel::BaseNode::SPtr(group->child.begin()->get()->toBaseNode()));

    simulation::CleanupVisitor cleanupvis(sofa::core::execparams::defaultInstance());
    cleanupvis.execute(group.get());
    simulation::DeleteVisitor vis(sofa::core::execparams::defaultInstance());
    vis.execute(group.get());
    group->detachFromGraph();
    group.reset();
}


void DefaultCollisionGroupManager::changeInstance(Instance inst)
{
    core::collision::CollisionGroupManager::changeInstance(inst);
    storedGroupSet[instance].swap(groupMap);
    groupMap.swap(storedGroupSet[inst]);
}

void DefaultCollisionGroupManager::createGroups(core::objectmodel::BaseContext* scene, const sofa::type::vector<Contact::SPtr>& contacts)
{
    SCOPED_TIMER("CreateGroups");

    int groupIndex = 1;

    // Map storing group merging history
    // key: node which has been moved into the node in value
    // value: node which received the content of the node in key
    MergeGroupsMap mergedGroups;

    // list of nodes that must be removed due to a move from a node to another
    sofa::type::vector< simulation::Node::SPtr > removedGroup;

    sofa::type::vector< std::pair<core::collision::Contact*, simulation::Node::SPtr> > contactGroup;

    //list of contacts considered stationary: one of the collision model has no associated ODE solver (= stationary)
    //for stationary contacts, no need to combine nodes into a single ODE solver
    sofa::type::vector< Contact* > stationaryContacts;

    for (const auto& contact : contacts)
    {
        createGroup(contact.get(), groupIndex, mergedGroups, contactGroup, removedGroup, stationaryContacts);
    }

    // create contact response for stationary contacts
    for (auto* contact : stationaryContacts)
    {
        contact->createResponse(scene);
    }

    // now that the groups are final, attach contacts' response
    // note: contact responses must be created in the same order than the contact list
    // This is to ensure the reproducibility of the simulation
    for (const auto& [contact, group] : contactGroup)
    {
        simulation::Node* g = getNodeFromMergedGroups(group.get(), mergedGroups);
        if (g != nullptr)
        {
            contact->createResponse(g);
        }
        else
        {
            contact->createResponse(scene);
        }
    }

    // delete removed groups
    for (auto& node : removedGroup)
    {
        node->detachFromGraph();
        node->execute<simulation::DeleteVisitor>(sofa::core::execparams::defaultInstance());
        node.reset();
    }
    removedGroup.clear();

    // finally recreate group vector
    groups.clear();
    for (auto& g : contactGroup)
    {
        if (g.second)
        {
            groups.push_back(g.second);
        }
    }
}

void DefaultCollisionGroupManager::createGroup(core::collision::Contact* contact,
                                               int& groupIndex,
                                               MergeGroupsMap& mergedGroups,
                                               sofa::type::vector< std::pair<core::collision::Contact*, simulation::Node::SPtr> >& contactGroup,
                                               sofa::type::vector< simulation::Node::SPtr >& removedGroup,
                                               sofa::type::vector< core::collision::Contact* >& stationaryContacts)
{
    const auto contactCollisionModels = contact->getCollisionModels();
    core::CollisionModel* cm_1 = contactCollisionModels.first;
    core::CollisionModel* cm_2 = contactCollisionModels.second;

    simulation::Node* group1 = getIntegrationNode(cm_1); //Node containing the ODE solver associated to cm_1
    simulation::Node* group2 = getIntegrationNode(cm_2); //Node containing the ODE solver associated to cm_1

    if (group1 == nullptr || group2 == nullptr)
    {
        // one of the object does not have an associated ODE solver
        // this can happen for stationary objects
        stationaryContacts.push_back(contact);
        return;
    }

    simulation::Node::SPtr collGroup = nullptr;
    if (group1 == group2)
    {
        // both collision models share the same group: no new group necessary
        collGroup = group1;
    }
    else if (simulation::Node* commonParent = group1->findCommonParent(group2))
    {
        const bool isSolverEmpty1 = group1->solver.empty();
        const bool isSolverEmpty2 = group2->solver.empty();

        // we can merge the groups
        // if solvers are compatible...
        const bool mergeSolvers = !isSolverEmpty1 && !isSolverEmpty2;
        SolverSet solver;
        if (mergeSolvers)
        {
            solver = SolverMerger::merge(*group1->solver.begin(), *group2->solver.begin());
        }

        if (!mergeSolvers || solver.odeSolver != nullptr)
        {
            auto group1Iter = groupMap.find(group1);
            auto group2Iter = groupMap.find(group2);

            const bool group1IsColl = group1Iter != groupMap.end();
            const bool group2IsColl = group2Iter != groupMap.end();

            if (!group1IsColl && !group2IsColl)
            {
                //none of the ODE solver nodes has been visited in the time step: create a new node
                const std::string childName { "collision" + std::to_string(groupIndex++) };

                collGroup = commonParent->createChild(childName);

                //move the first ODE solver node into the new node
                collGroup->moveChild(BaseNode::SPtr(group1));

                //move the second ODE solver node into the new node
                collGroup->moveChild(BaseNode::SPtr(group2));

                groupMap[group1] = collGroup.get();
                groupMap[group2] = collGroup.get();
                groupMap[collGroup.get()] = collGroup.get();
            }
            else if (group1IsColl) //the first ODE solver node has been visited during this time step
            {
                collGroup = getNodeFromMergedGroups(group1Iter->second, mergedGroups);

                // merge group2 in group1
                if (!group2IsColl) //the second ODE solver node has NOT been visited during this time step
                {
                    //second ODE solver node is moved into the first ODE solver node
                    collGroup->moveChild(BaseNode::SPtr(group2));
                    groupMap[group2] = collGroup.get();
                }
                else //the second ODE solver node has been visited during this time step
                {
                    simulation::Node::SPtr collGroup2 = getNodeFromMergedGroups(group2Iter->second, mergedGroups);
                    if (collGroup == collGroup2)
                    {
                        // both ODE solver nodes are already in the same collision group
                        groupMap[group1Iter->first] = collGroup.get();
                        groupMap[group2Iter->first] = collGroup.get();
                        contactGroup.emplace_back(contact, collGroup);
                        return;
                    }
                    else
                    {
                        //both ODE solver nodes have been visited, but they are not in the same node
                        //move the second node into the first
                        //solvers of the second node are deleted

                        groupMap[group2] = collGroup.get();
                        // merge groups and remove collGroup2

                        // store solvers of the second group and destroy them later
                        SolverSet tmpSolverSet;
                        if (mergeSolvers)
                        {
                            getSolverSet(collGroup2, tmpSolverSet);

                            //remove solvers from group2, so it is not moved later with the rest of the objects
                            removeSolverSetFromNode(collGroup2, tmpSolverSet);
                        }
                        //move all objects from group 2 to group 1
                        moveAllObjects(collGroup2, collGroup);
                        moveAllChildren(collGroup2, collGroup);

                        //group2 is empty: remove the node
                        commonParent->removeChild(collGroup2);

                        //group is added to a list of groups that will be destroyed later
                        removedGroup.push_back(collGroup2);
                        groupMap.erase(collGroup2.get());

                        //stores that group2 has been merged into group1
                        mergedGroups[collGroup2.get()] = collGroup.get();

                        //solvers object can be safely destroyed
                        destroySolvers(tmpSolverSet);
                    }
                }
            }
            else //only the second ODE solver node has been visited during this time step
            {
                collGroup = getNodeFromMergedGroups(group2Iter->second, mergedGroups);
                // group1 is not a collision group while group2 is
                collGroup->moveChild(BaseNode::SPtr(group1));
                groupMap[group1] = collGroup.get();
            }

            SolverSet tmpSolverSet;
            getSolverSet(collGroup, tmpSolverSet);
            removeSolverSetFromNode(collGroup, tmpSolverSet);

            addSolversToNode(collGroup, solver);
        }
    }
    contactGroup.emplace_back(contact, collGroup);
}


void DefaultCollisionGroupManager::clearGroups(core::objectmodel::BaseContext* /*scene*/)
{
    for (const auto& nodePair : groupMap)
    {
        if (nodePair.second != nullptr && !nodePair.second->getParents().empty())
        {
            clearCollisionGroup(nodePair.second);
        }
    }

    groupMap.clear();
    groups.clear();
}


simulation::Node* DefaultCollisionGroupManager::getIntegrationNode(core::CollisionModel* model)
{
    simulation::Node* node = static_cast<simulation::Node*>(model->getContext());
    type::vector< core::behavior::OdeSolver *> listSolver;
    node->get< core::behavior::OdeSolver >(&listSolver);

    if (listSolver.empty())
        return nullptr;

    simulation::Node* solvernode = static_cast<simulation::Node*>(listSolver.back()->getContext());
    return solvernode;
}

void DefaultCollisionGroupManager::moveAllObjects(simulation::Node::SPtr sourceNode, simulation::Node::SPtr destinationNode)
{
    while(!sourceNode->object.empty())
    {
        destinationNode->moveObject(*sourceNode->object.begin());
    }
}

void DefaultCollisionGroupManager::moveAllChildren(simulation::Node::SPtr sourceNode, simulation::Node::SPtr destinationNode)
{
    while(!sourceNode->child.empty())
    {
        destinationNode->moveChild(*sourceNode->child.begin());
    }
}

void DefaultCollisionGroupManager::getSolverSet(simulation::Node::SPtr node, SolverSet& solverSet)
{
    if (!node->linearSolver.empty())
    {
        solverSet.odeSolver = *node->solver.begin();
    }

    if (!node->linearSolver.empty())
    {
        solverSet.linearSolver = *node->linearSolver.begin();
    }

    if (!node->constraintSolver.empty())
    {
        solverSet.constraintSolver = *node->constraintSolver.begin();
    }
}

void DefaultCollisionGroupManager::removeSolverSetFromNode(simulation::Node::SPtr node, sofa::component::collision::SolverSet& solverSet)
{
    if (solverSet.odeSolver)
    {
        node->removeObject(solverSet.odeSolver);
    }
    if (solverSet.linearSolver)
    {
        node->removeObject(solverSet.linearSolver);
    }
    if (solverSet.constraintSolver)
    {
        node->removeObject(solverSet.constraintSolver);
    }
}

void DefaultCollisionGroupManager::destroySolvers(sofa::component::collision::SolverSet& solverSet)
{
    if (solverSet.odeSolver != nullptr)
    {
        solverSet.odeSolver.reset();
    }
    if (solverSet.linearSolver != nullptr)
    {
        solverSet.linearSolver.reset();
    }
    if (solverSet.constraintSolver != nullptr)
    {
        solverSet.constraintSolver.reset();
    }
}

void DefaultCollisionGroupManager::addSolversToNode(simulation::Node::SPtr node, sofa::component::collision::SolverSet& solverSet)
{
    if (solverSet.odeSolver)
    {
        node->addObject(solverSet.odeSolver);
    }
    if (solverSet.linearSolver)
    {
        node->addObject(solverSet.linearSolver);
    }
    if (solverSet.constraintSolver)
    {
        node->addObject(solverSet.constraintSolver);
    }
    // perform init only once everyone has been added (in case of explicit dependencies)
    if (solverSet.odeSolver)
    {
        solverSet.odeSolver->init();
    }
    if (solverSet.linearSolver)
    {
        solverSet.linearSolver->init();
    }
    if (solverSet.constraintSolver)
    {
        solverSet.constraintSolver->init();
    }
}

simulation::Node* DefaultCollisionGroupManager::getNodeFromMergedGroups(simulation::Node* node, const MergeGroupsMap& mergedGroups)
{
    if (node != nullptr)
    {
        simulation::Node* nodeBegin = node;
        auto it = mergedGroups.find(node);

        //the group associated to a contact may have been moved to another group
        while (it != mergedGroups.cend() && it->second != nullptr)
        {
            node = it->second;
            checkEndlessLoop(mergedGroups, nodeBegin, node);
            it = mergedGroups.find(node);
        }
    }
    return node;
}

void DefaultCollisionGroupManager::checkEndlessLoop(const MergeGroupsMap& mergedGroups, simulation::Node* firstGroup, simulation::Node* currentGroup)
{
    if (currentGroup == firstGroup)
    {
        std::stringstream msg;
        msg << "A logic problem (endless loop) has been detected. Please report a bug on https://github.com/sofa-framework/sofa/issues. Details:\n";
        for (const auto& [a, b] : mergedGroups)
        {
            msg << a << "(" << (a ? a->getPathName() : "invalid") << ") " << b << "(" << (b ? b->getPathName() : "invalid") << "), ";
        }

        msg_fatal() << msg.str();
        throw std::logic_error(msg.str());
    }
}

} // namespace sofa::component::collision
