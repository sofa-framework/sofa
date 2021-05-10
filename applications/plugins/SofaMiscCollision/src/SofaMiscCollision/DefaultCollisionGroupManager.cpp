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
#include <SofaMiscCollision/SolverMerger.h>
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

DefaultCollisionGroupManager::DefaultCollisionGroupManager()= default;
DefaultCollisionGroupManager::~DefaultCollisionGroupManager()= default;

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

void DefaultCollisionGroupManager::createGroups(core::objectmodel::BaseContext* scene, const sofa::helper::vector<Contact::SPtr>& contacts)
{
    sofa::helper::ScopedAdvancedTimer timer("CreateGroups");

    int groupIndex = 1;

    // Map storing group merging history
    std::map<simulation::Node*, simulation::Node::SPtr > mergedGroups;
    sofa::helper::vector< simulation::Node::SPtr > removedGroup;
    std::map<Contact*, simulation::Node::SPtr> contactGroup;

    for (const auto& contact : contacts)
    {
        createGroup(contact.get(), groupIndex, mergedGroups, contactGroup, removedGroup);
    }

    // now that the groups are final, attach contacts' response
    for (const auto& [contact, g] : contactGroup)
    {
        simulation::Node::SPtr group = g;
        while (group != nullptr && mergedGroups.find(group.get()) != mergedGroups.end())
            group = mergedGroups[group.get()];
        if (group!=nullptr)
            contact->createResponse(group.get());
        else
            contact->createResponse(scene);
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
    std::transform(contactGroup.begin(), contactGroup.end(), std::back_inserter(groups), [](const auto& g){return g.second;});
}

void DefaultCollisionGroupManager::createGroup(core::collision::Contact* contact,
                                               int& groupIndex,
                                               std::map<simulation::Node*, simulation::Node::SPtr >& mergedGroups,
                                               std::map<core::collision::Contact*, simulation::Node::SPtr>& contactGroup,
                                               sofa::helper::vector< simulation::Node::SPtr >& removedGroup)
{
    core::CollisionModel* cm_1 = contact->getCollisionModels().first;
    core::CollisionModel* cm_2 = contact->getCollisionModels().second;

    simulation::Node* group1 = getIntegrationNode(cm_1); //Node containing the ODE solver associated to cm_1
    simulation::Node* group2 = getIntegrationNode(cm_2); //Node containing the ODE solver associated to cm_1

    if (group1 == nullptr || group2 == nullptr)
    {
        // one of the object does not have an associated ODE solver
        // this can happen for stationary objects
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
        // we can merge the groups
        // if solvers are compatible...
        const bool mergeSolvers = !group1->solver.empty() && !group2->solver.empty();
        SolverSet solver;
        if (mergeSolvers)
        {
            solver = SolverMerger::merge(group1->solver[0], group2->solver[0]);
        }


        if (!mergeSolvers || solver.odeSolver!=nullptr)
        {
            auto group1Iter = groupMap.find(group1);
            auto group2Iter = groupMap.find(group2);
            const bool group1IsColl = group1Iter != groupMap.end();
            const bool group2IsColl = group2Iter != groupMap.end();
            if (!group1IsColl && !group2IsColl)
            {
                //none of the ODE solver nodes has been visited in the time step: create a new node
                collGroup = commonParent->createChild("collision" + std::to_string(groupIndex++));

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
                collGroup = group1Iter->second;

                // merge group2 in group1
                if (!group2IsColl) //the second ODE solver node has NOT been visited during this time step
                {
                    //second ODE solver node is moved into the first ODE solver node
                    collGroup->moveChild(BaseNode::SPtr(group2));
                }
                else //the second ODE solver node has been visited during this time step
                {
                    simulation::Node::SPtr collGroup2 = group2Iter->second;
                    if (collGroup == collGroup2)
                    {
                        // both ODE solver nodes are already in the same collision group
                        groupMap[group1Iter->first] = group1Iter->second;
                        groupMap[group2Iter->first] = group2Iter->second;
                        contactGroup[contact] = collGroup;
                        return;
                    }

                    //both ODE solver nodes have been visited, but they are not in the same node

                    groupMap[group2] = collGroup.get();
                    // merge groups and remove collGroup2
                    SolverSet solver2;
                    if (mergeSolvers)
                    {
                        solver2.odeSolver = collGroup2->solver[0];
                        collGroup2->removeObject(solver2.odeSolver);
                        if (!collGroup2->linearSolver.empty())
                        {
                            solver2.linearSolver = collGroup2->linearSolver[0];
                            collGroup2->removeObject(solver2.linearSolver);
                        }
                        if (!collGroup2->constraintSolver.empty())
                        {
                            solver2.constraintSolver = collGroup2->constraintSolver[0];
                            collGroup2->removeObject(solver2.constraintSolver);
                        }
                    }
                    while(!collGroup2->object.empty())
                        collGroup->moveObject(*collGroup2->object.begin());
                    while(!collGroup2->child.empty())
                        collGroup->moveChild(BaseNode::SPtr(*collGroup2->child.begin()));
                    commonParent->removeChild(collGroup2);
                    groupMap.erase(collGroup2.get());
                    mergedGroups[collGroup2.get()] = collGroup;
                    if (solver2.odeSolver) solver2.odeSolver.reset();
                    if (solver2.linearSolver) solver2.linearSolver.reset();
                    if (solver2.constraintSolver) solver2.constraintSolver.reset();
                    // BUGFIX(2007-06-23 Jeremie A): we can't remove group2 yet, to make sure the keys in mergedGroups are unique.
                    removedGroup.push_back(collGroup2);
                }
            }
            else
            {
                collGroup = group2Iter->second;
                // group1 is not a collision group while group2 is
                collGroup->moveChild(BaseNode::SPtr(group1));
                groupMap[group1] = collGroup.get();
            }
            if (!collGroup->solver.empty())
            {
                core::behavior::OdeSolver* solver2 = collGroup->solver[0];
                collGroup->removeObject(solver2);
            }
            if (!collGroup->linearSolver.empty())
            {
                core::behavior::BaseLinearSolver* solver2 = collGroup->linearSolver[0];
                collGroup->removeObject(solver2);
            }
            if (!collGroup->constraintSolver.empty())
            {
                core::behavior::ConstraintSolver* solver2 = collGroup->constraintSolver[0];
                collGroup->removeObject(solver2);
            }
            if (solver.odeSolver)
            {
                collGroup->addObject(solver.odeSolver);
            }
            if (solver.linearSolver)
            {
                collGroup->addObject(solver.linearSolver);
            }
            if (solver.constraintSolver)
            {
                collGroup->addObject(solver.constraintSolver);
            }
            // perform init only once everyone has been added (in case of explicit dependencies)
            if (solver.odeSolver)
            {
                solver.odeSolver->init();
            }
            if (solver.linearSolver)
            {
                solver.linearSolver->init();
            }
            if (solver.constraintSolver)
            {
                solver.constraintSolver->init();
            }
        }
    }
    contactGroup[contact] = collGroup;
}


void DefaultCollisionGroupManager::clearGroups(core::objectmodel::BaseContext* /*scene*/)
{
    for (const auto& nodePair : groupMap)
    {
        if (!nodePair.second->getParents().empty())
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
    helper::vector< core::behavior::OdeSolver *> listSolver;
    node->get< core::behavior::OdeSolver >(&listSolver);

    if (listSolver.empty())
        return nullptr;

    simulation::Node* solvernode = static_cast<simulation::Node*>(listSolver.back()->getContext());
    return solvernode;
}

} // namespace sofa::component::collision
