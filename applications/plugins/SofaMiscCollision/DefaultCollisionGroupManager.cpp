/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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


SOFA_DECL_CLASS(DefaultCollisionGroupManager);

int DefaultCollisionGroupManagerClass = core::RegisterObject("Responsible for gathering colliding objects in the same group, for consistent time integration")
        .add< DefaultCollisionGroupManager >()
        .addAlias( "CollisionGroupManager" )
        .addAlias( "CollisionGroup" )
        .addAlias( "TreeCollisionGroupManager" ) // for backward compatibility with old scene files but could be removed
        ;




DefaultCollisionGroupManager::DefaultCollisionGroupManager()
{
}

DefaultCollisionGroupManager::~DefaultCollisionGroupManager()
{
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
        simulation::Node::SPtr group = NULL;
        if (group1==NULL || group2==NULL)
        {
        }
        else if (group1 == group2)
        {
            // same group, no new group necessary
            group = group1;
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
                bool group1IsColl = groupSet.find(group1)!=groupSet.end();
                bool group2IsColl = groupSet.find(group2)!=groupSet.end();
                if (!group1IsColl && !group2IsColl)
                {
                    char groupName[32];
                    snprintf(groupName,sizeof(groupName),"collision%d",groupIndex++);
                    // create a new group
                    group = parent->createChild(groupName);

                    group->moveChild((simulation::Node*)group1);
                    group->moveChild((simulation::Node*)group2);
                    groupSet.insert(group.get());
                }
                else if (group1IsColl)
                {
                    group = group1;
                    // merge group2 in group1
                    if (!group2IsColl)
                    {
                        group->moveChild(group2);
                    }
                    else
                    {
                        // merge groups and remove group2
                        SolverSet solver2;
                        if (mergeSolvers)
                        {
                            solver2.odeSolver = group2->solver[0];
                            group2->removeObject(solver2.odeSolver);
                            if (!group2->linearSolver.empty())
                            {
                                solver2.linearSolver = group2->linearSolver[0];
                                group2->removeObject(solver2.linearSolver);
                            }
                            if (!group2->constraintSolver.empty())
                            {
                                solver2.constraintSolver = group2->constraintSolver[0];
                                group2->removeObject(solver2.constraintSolver);
                            }
                        }
                        while(!group2->object.empty())
                            group->moveObject(*group2->object.begin());
                        while(!group2->child.empty())
                            group->moveChild(*group2->child.begin());
                        parent->removeChild((simulation::Node*)group2);
                        groupSet.erase(group2);
                        mergedGroups[group2] = group;
                        if (solver2.odeSolver) solver2.odeSolver.reset();
                        if (solver2.linearSolver) solver2.linearSolver.reset();
                        if (solver2.constraintSolver) solver2.constraintSolver.reset();
                        // BUGFIX(2007-06-23 Jeremie A): we can't remove group2 yet, to make sure the keys in mergedGroups are unique.
                        removedGroup.push_back(group2);
                        //delete group2;
                    }
                }
                else
                {
                    // group1 is not a collision group while group2 is
                    group = group2;
                    group->moveChild(group1);
                }
                if (!group->solver.empty())
                {
                    core::behavior::OdeSolver* solver2 = group->solver[0];
                    group->removeObject(solver2);
                }
                if (!group->linearSolver.empty())
                {
                    core::behavior::BaseLinearSolver* solver2 = group->linearSolver[0];
                    group->removeObject(solver2);
                }
                if (!group->constraintSolver.empty())
                {
                    core::behavior::ConstraintSolver* solver2 = group->constraintSolver[0];
                    group->removeObject(solver2);
                }
                if (solver.odeSolver)
                {
                    group->addObject(solver.odeSolver);
                }
                if (solver.linearSolver)
                {
                    group->addObject(solver.linearSolver);
                }
                if (solver.constraintSolver)
                {
                    group->addObject(solver.constraintSolver);
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
        contactGroup.push_back(group);
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
    for (std::set<simulation::Node::SPtr>::iterator it = groupSet.begin(); it!=groupSet.end(); ++it)
        groups.push_back(*it);
}


void DefaultCollisionGroupManager::clearGroups(core::objectmodel::BaseContext* /*scene*/)
{
    for (std::set<simulation::Node::SPtr>::iterator it = groupSet.begin(); it!=groupSet.end(); ++it)
    {
        if (*it) clearGroup( (*it)->getParents(), *it );
    }

    groupSet.clear();
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
