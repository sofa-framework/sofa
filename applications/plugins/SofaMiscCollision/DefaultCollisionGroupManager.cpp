/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/simulation/Simulation.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace collision
{

using core::collision::Contact;
using core::objectmodel::BaseNode;


int DefaultCollisionGroupManagerClass = core::RegisterObject("Responsible for gathering colliding objects in the same group, for consistent time integration")
        .add< DefaultCollisionGroupManager >()
        .addAlias( "CollisionGroupManager" )
        .addAlias( "CollisionGroup" )
        .addAlias( "TreeCollisionGroupManager" ) // for backward compatibility with old scene files but could be removed
        ;

DefaultCollisionGroupManager::DefaultCollisionGroupManager(){}
DefaultCollisionGroupManager::~DefaultCollisionGroupManager(){}

void DefaultCollisionGroupManager::clearCollisionGroup(simulation::Node::SPtr group)
{
    const simulation::Node::Parents &parents = group->getParents();
    core::objectmodel::BaseNode::SPtr parent = *parents.begin();
    while (!group->child.empty()) 
        parent->moveChild(core::objectmodel::BaseNode::SPtr(group->child.begin()->get()->toBaseNode()));

    simulation::CleanupVisitor cleanupvis(sofa::core::ExecParams::defaultInstance());
    cleanupvis.execute(group.get());
    simulation::DeleteVisitor vis(sofa::core::ExecParams::defaultInstance());
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
    int groupIndex = 1;

    // Map storing group merging history
    std::map<simulation::Node*, simulation::Node::SPtr > mergedGroups;
    sofa::helper::vector< simulation::Node::SPtr > contactGroup;
    sofa::helper::vector< simulation::Node::SPtr > removedGroup;
    contactGroup.reserve(contacts.size());
    for(sofa::helper::vector<Contact::SPtr>::const_iterator cit = contacts.begin(); cit != contacts.end(); ++cit)
    {
        Contact* contact = cit->get();
        simulation::Node* group1 = getIntegrationNode(contact->getCollisionModels().first);
        simulation::Node* group2 = getIntegrationNode(contact->getCollisionModels().second);
        simulation::Node::SPtr collGroup = NULL;
        if (group1==NULL || group2==NULL)
        {
        }
        else if (group1 == group2)
        {
            // same group, no new group necessary
            collGroup = group1;
        }
        else if (simulation::Node* parent=group1->findCommonParent(group2))
        {
            // we can merge the groups
            // if solvers are compatible...
            bool mergeSolvers = (!group1->solver.empty() || !group2->solver.empty());
            SolverSet solver;
            if (mergeSolvers)
                solver = SolverMerger::merge(group1->solver[0], group2->solver[0]);


            if (!mergeSolvers || solver.odeSolver!=NULL)
            {
                auto group1Iter = groupMap.find(group1);
                auto group2Iter = groupMap.find(group2);
                bool group1IsColl = group1Iter != groupMap.end();
                bool group2IsColl = group2Iter != groupMap.end();
                if (!group1IsColl && !group2IsColl)
                {
                    char groupName[32];
                    snprintf(groupName,sizeof(groupName),"collision%d",groupIndex++);
                    // create a new group
                    collGroup = parent->createChild(groupName);

                    collGroup->moveChild(BaseNode::SPtr(group1));
                    collGroup->moveChild(BaseNode::SPtr(group2));
                    groupMap[group1] = collGroup.get();
                    groupMap[group2] = collGroup.get();
                }
                else if (group1IsColl)
                {
                    collGroup = group1Iter->second;
                    // merge group2 in group1
                    if (!group2IsColl)
                    {
                        collGroup->moveChild(BaseNode::SPtr(group2));
                    }
                    else
                    {
                        simulation::Node::SPtr collGroup2 = group2Iter->second;
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
                        parent->removeChild(collGroup2);
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
        contactGroup.push_back(collGroup);
    }

    // now that the groups are final, attach contacts' response
    for(unsigned int i=0; i<contacts.size(); i++)
    {
        Contact* contact = contacts[i].get();
        simulation::Node::SPtr group = contactGroup[i];
        while (group!=NULL && mergedGroups.find(group.get())!=mergedGroups.end())
            group = mergedGroups[group.get()];
        if (group!=NULL)
            contact->createResponse(group.get());
        else
            contact->createResponse(scene);
    }

    // delete removed groups
    for (sofa::helper::vector<simulation::Node::SPtr>::iterator it = removedGroup.begin(); it!=removedGroup.end(); ++it)
    {
        simulation::Node::SPtr node = *it;
        node->detachFromGraph();
        node->execute<simulation::DeleteVisitor>(sofa::core::ExecParams::defaultInstance());
        it->reset();
    }
    removedGroup.clear();

    // finally recreate group vector
    groups.clear();
    for (auto it = contactGroup.begin(); it!= contactGroup.end(); ++it)
        groups.push_back(*it);
}


void DefaultCollisionGroupManager::clearGroups(core::objectmodel::BaseContext* /*scene*/)
{
    for (std::map<simulation::Node*, simulation::Node*>::iterator it = groupMap.begin(); it!=groupMap.end(); ++it)
    {
        if (it->second->getParents().size() > 0)
        {
            clearCollisionGroup(it->second);
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
        return NULL;
    simulation::Node* solvernode = static_cast<simulation::Node*>(listSolver.back()->getContext());
    if (solvernode->linearSolver.empty())
        return solvernode; // no linearsolver
    core::behavior::BaseLinearSolver * linearSolver = solvernode->linearSolver[0];
    if (!linearSolver->isMultiGroup())
    {
        return solvernode;
    }
    // This solver handles multiple groups, we have to find which group contains this collision model
    // First move up to the node of the initial mechanical object
    while (node->mechanicalMapping && node->mechanicalMapping->getMechFrom()[0])
        node = static_cast<simulation::Node*>(node->mechanicalMapping->getMechFrom()[0]->getContext());

    // Then check if it is one of the child nodes of the solver node
    for (simulation::Node::ChildIterator it = solvernode->child.begin(), itend = solvernode->child.end(); it != itend; ++it)
        if (*it == node)
        {
            return it->get();
        }

    // Then check if it is a child of one of the child nodes of the solver node
    for (simulation::Node::ChildIterator it = solvernode->child.begin(), itend = solvernode->child.end(); it != itend; ++it)
        if (node->hasParent(it->get()))
            return it->get();

    // Then check if it is a grand-childs of one of the child nodes of the solver node
    for (simulation::Node::ChildIterator it = solvernode->child.begin(), itend = solvernode->child.end(); it != itend; ++it)
        if (node->getContext()->hasAncestor(it->get()))
            return it->get();

    // group not found, simply return the solver node
    return solvernode;
}

} // namespace collision

} // namespace component

} // namespace Sofa
