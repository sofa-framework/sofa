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
#include "BglCollisionGroupManager.h"
#include <sofa/component/collision/SolverMerger.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/CollisionModel.h>
// #include <sofa/helper/system/config.h>
// #include <string.h>


#include "BglNode.h"
#include <sofa/simulation/common/Simulation.h>

namespace sofa
{

namespace component
{

namespace collision
{

using core::componentmodel::collision::Contact;

SOFA_DECL_CLASS(BglCollisionGroupManager);

int BglCollisionGroupManagerClass = core::RegisterObject("Responsible for gathering colliding objects in the same group, for consistent time integration")
        .add< BglCollisionGroupManager >()
        ;


BglCollisionGroupManager::BglCollisionGroupManager()
{
}

BglCollisionGroupManager::~BglCollisionGroupManager()
{
}

simulation::Node* BglCollisionGroupManager::buildCollisionGroup()
{
    return simulation::bgl::getSimulation()->newNode("CollisionGroup");
}


template <typename ContainerParent>
typename ContainerParent::value_type BglCollisionGroupManager::compatibleSetOfNode( ContainerParent &set1,ContainerParent &set2)
{
    typename ContainerParent::iterator
    it1, it1_end=set1.end(),
         it2, it2_end=set2.end();
    for (it1=set1.begin(); it1!=it1_end; ++it1)
    {
        for (it2=set2.begin(); it2!=it2_end; ++it2)
        {
            if (*it1 == *it2)
            {
                return *it1;
            }
        }
    }
    return NULL;
}

void BglCollisionGroupManager::createGroups(core::objectmodel::BaseContext* scene, const sofa::helper::vector<Contact*>& contacts)
{
    int groupIndex = 1;
    simulation::Node* node = dynamic_cast<simulation::Node*>(scene);
    if (node==NULL)
    {
        serr << "BglCollisionGroupManager only support graph-based scenes."<<sendl;
        return;
    }

    if (node && !node->getLogTime()) node=NULL; // Only use node for time logging
    simulation::Node::ctime_t t0 = 0;

    if (node) t0 = node->startTime();

    // Map storing group merging history
    std::map<simulation::Node*, simulation::Node*> mergedGroups;
    sofa::helper::vector<simulation::Node*> contactGroup;
    sofa::helper::vector<simulation::Node*> removedGroup;
    contactGroup.reserve(contacts.size());
    for(sofa::helper::vector<Contact*>::const_iterator cit = contacts.begin(); cit != contacts.end(); cit++)
    {
        Contact* contact = *cit;
        simulation::bgl::BglNode* group1 = static_cast<simulation::bgl::BglNode*>(getIntegrationNode(contact->getCollisionModels().first));
        simulation::bgl::BglNode* group2 = static_cast<simulation::bgl::BglNode*>(getIntegrationNode(contact->getCollisionModels().second));

        simulation::Node* group = NULL;
        if (group1==NULL || group2==NULL)
        {
        }
        else if (group1 == group2)
        {
            // same group, no new group necessary
            group = group1;
        }
        else
        {
            simulation::bgl::BglNode::Parents parents1=group1->getParents();
            if (!parents1.empty())
            {
                simulation::bgl::BglNode::Parents parents2=group2->getParents();
                simulation::bgl::BglNode* commonParent=compatibleSetOfNode(parents1,parents2);

                if (commonParent)
                {
                    // we can merge the groups
                    // if solvers are compatible...
                    SolverSet solver = SolverMerger::merge(group1->solver[0], group2->solver[0]);
                    if (solver.first!=NULL)
                    {
                        simulation::bgl::BglNode* parent = commonParent;
                        bool group1IsColl = groupSet.find(group1)!=groupSet.end();
                        bool group2IsColl = groupSet.find(group2)!=groupSet.end();
                        if (!group1IsColl && !group2IsColl)
                        {
                            char groupName[32];
                            snprintf(groupName,sizeof(groupName),"collision%d",groupIndex++);
                            // create a new group
                            group = buildCollisionGroup();
                            group->setName(groupName);
                            parent->addChild(group);

                            core::objectmodel::Context *current_context = dynamic_cast< core::objectmodel::Context *>(parent->getContext());
                            group->copyVisualContext( (*current_context));

                            group->updateSimulationContext();
                            group->moveChild((simulation::Node*)group1);
                            group->moveChild((simulation::Node*)group2);
                            groupSet.insert(group);
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
                                solver2.first = group2->solver[0];
                                group2->removeObject(solver2.first);
                                if (!group2->linearSolver.empty())
                                {
                                    solver2.second = group2->linearSolver[0];
                                    group2->removeObject(solver2.second);
                                }
                                else
                                    solver2.second = NULL;
                                while(!group2->object.empty())
                                    group->moveObject(*group2->object.begin());
                                while(!group2->child.empty())
                                    group->moveChild(*group2->child.begin());
                                parent->removeChild((simulation::Node*)group2);
                                groupSet.erase(group2);
                                mergedGroups[group2] = group;
                                delete solver2.first;
                                if (solver2.second) delete solver2.second;
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
                            core::componentmodel::behavior::OdeSolver* solver2 = group->solver[0];
                            group->removeObject(solver2);
                            delete solver2;
                        }
                        if (!group->linearSolver.empty())
                        {
                            core::componentmodel::behavior::LinearSolver* solver2 = group->linearSolver[0];
                            group->removeObject(solver2);
                            delete solver2;
                        }
                        group->addObject(solver.first);
                        if (solver.second)
                            group->addObject(solver.second);
                    }
                }
            }
        }
        contactGroup.push_back(group);
    }

    if (node) t0 = node->endTime(t0, "collision/groups", this);

    // now that the groups are final, attach contacts' response
    for(unsigned int i=0; i<contacts.size(); i++)
    {
        Contact* contact = contacts[i];
        simulation::Node* group = contactGroup[i];
        while (group!=NULL && mergedGroups.find(group)!=mergedGroups.end())
            group = mergedGroups[group];
        if (group!=NULL)
            contact->createResponse(group);
        else
            contact->createResponse(scene);
    }

    if (node) t0 = node->endTime(t0, "collision/contacts", this);

    // delete removed groups
    for (sofa::helper::vector<simulation::Node*>::iterator it = removedGroup.begin(); it!=removedGroup.end(); ++it)
        delete *it;
    removedGroup.clear();

    // finally recreate group vector
    groups.clear();
    for (std::set<simulation::Node*>::iterator it = groupSet.begin(); it!=groupSet.end(); ++it)
        groups.push_back(*it);
    //if (!groups.empty())
    //	sout << groups.size()<<" collision groups created."<<sendl;
}

//Functor to remove the child from all the parents node
struct childCleaner
{
    childCleaner(simulation::Node* n):node(n) {};

    void operator()(simulation::bgl::BglNode *bglNode)
    {
        bglNode->removeChild(node);
    }

    simulation::Node *node;
};


void BglCollisionGroupManager::clearGroups(core::objectmodel::BaseContext* /*scene*/)
{
    for (std::set<simulation::Node*>::iterator it = groupSet.begin(); it!=groupSet.end(); ++it)
    {
        simulation::bgl::BglNode* group = dynamic_cast<simulation::bgl::BglNode*>(*it);
        if (group)
        {
            simulation::bgl::BglNode::Parents parents=group->getParents();
            simulation::bgl::BglNode* parent = *parents.begin();
            while(!group->child.empty())
                parent->moveChild(*group->child.begin());
            while(!group->object.empty())
            {
                core::objectmodel::BaseObject* obj = *group->object.begin();
                group->removeObject(obj);
                delete obj;
            }

            std::for_each(parents.begin(),parents.end(), childCleaner((simulation::Node*)group));
            delete group;
        }
    }

    groupSet.clear();
    groups.clear();
}

simulation::Node* BglCollisionGroupManager::getIntegrationNode(core::CollisionModel* model)
{
    simulation::Node* node = static_cast<simulation::Node*>(model->getContext());

    helper::vector< core::componentmodel::behavior::OdeSolver *> listSolver;
    node->get< core::componentmodel::behavior::OdeSolver >(&listSolver);

    if (!listSolver.empty()) return static_cast<simulation::Node*>(listSolver.back()->getContext());
    else                     return NULL;

}

}// namespace collision

} // namespace component

} // namespace Sofa
