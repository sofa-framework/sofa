/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/component/collision/DefaultCollisionGroupManager.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/helper/system/config.h>
#include <sofa/helper/FnDispatcher.h>
#include <sofa/helper/FnDispatcher.inl>
#include <sofa/component/collision/DefaultCollisionGroupManager.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/component/odesolver/RungeKutta4Solver.h>
#include <sofa/component/odesolver/CGImplicitSolver.h>
#include <sofa/core/ObjectFactory.h>
#include <string.h>



namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace core::componentmodel::behavior;
using namespace core::componentmodel::collision;

SOFA_DECL_CLASS(DefaultCollisionGroupManager)

int DefaultCollisionGroupManagerClass = core::RegisterObject("Responsible for gathering colliding objects in the same group, for consistent time integration")
        .add< DefaultCollisionGroupManager >()
        .addAlias("CollisionGroup")
        ;

class SolverMerger
{
public:
    static core::componentmodel::behavior::OdeSolver* merge(core::componentmodel::behavior::OdeSolver* solver1, core::componentmodel::behavior::OdeSolver* solver2);

protected:

    FnDispatcher<core::componentmodel::behavior::OdeSolver, core::componentmodel::behavior::OdeSolver*> solverDispatcher;

    SolverMerger ();
};

DefaultCollisionGroupManager::DefaultCollisionGroupManager()
{
}

DefaultCollisionGroupManager::~DefaultCollisionGroupManager()
{
}

void DefaultCollisionGroupManager::createGroups(core::objectmodel::BaseContext* scene, const std::vector<Contact*>& contacts)
{
    int groupIndex = 1;
    simulation::tree::GNode* groot = dynamic_cast<simulation::tree::GNode*>(scene);
    if (groot==NULL)
    {
        std::cerr << "DefaultCollisionGroupManager only support graph-based scenes.\n";
        return;
    }

    simulation::tree::GNode* node = groot;
    if (node && !node->getLogTime()) node=NULL; // Only use node for time logging
    simulation::tree::GNode::ctime_t t0 = 0;

    if (node) t0 = node->startTime();

    // Map storing group merging history
    std::map<simulation::tree::GNode*, simulation::tree::GNode*> mergedGroups;
    std::vector<simulation::tree::GNode*> contactGroup;
    std::vector<simulation::tree::GNode*> removedGroup;
    contactGroup.reserve(contacts.size());
    for(std::vector<Contact*>::const_iterator cit = contacts.begin(); cit != contacts.end(); cit++)
    {
        Contact* contact = *cit;
        simulation::tree::GNode* group1 = getIntegrationNode(contact->getCollisionModels().first);
        simulation::tree::GNode* group2 = getIntegrationNode(contact->getCollisionModels().second);
        simulation::tree::GNode* group = NULL;
        if (group1==NULL || group2==NULL)
        {
        }
        else if (group1 == group2)
        {
            // same group, no new group necessary
            group = group1;
        }
        else if (group1->getParent()!=NULL && group1->getParent() == group2->getParent())
        {
            // we can merge the groups
            // if solvers are compatible...
            OdeSolver* solver = SolverMerger::merge(group1->solver, group2->solver);
            if (solver!=NULL)
            {
                simulation::tree::GNode* parent = group1->parent;
                bool group1IsColl = groupSet.find(group1)!=groupSet.end();
                bool group2IsColl = groupSet.find(group2)!=groupSet.end();
                if (!group1IsColl && !group2IsColl)
                {
                    char groupName[32];
                    snprintf(groupName,sizeof(groupName),"collision%d",groupIndex++);
                    // create a new group
                    group = new simulation::tree::GNode(groupName);
                    parent->addChild(group);
                    group->updateContext();
                    group->moveChild(group1);
                    group->moveChild(group2);
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
                        OdeSolver* solver2 = group2->solver;
                        group2->removeObject(solver2);
                        while(!group2->object.empty())
                            group->moveObject(*group2->object.begin());
                        while(!group2->child.empty())
                            group->moveChild(*group2->child.begin());
                        parent->removeChild(group2);
                        groupSet.erase(group2);
                        mergedGroups[group2] = group;
                        delete solver2;
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
                if (group->solver!=NULL)
                {
                    OdeSolver* solver2 = group->solver;
                    group->removeObject(solver2);
                    delete solver2;
                }
                group->addObject(solver);
            }
        }
        contactGroup.push_back(group);
    }

    if (node) t0 = node->endTime(t0, "collision/groups", this);

    // now that the groups are final, attach contacts' response
    for(unsigned int i=0; i<contacts.size(); i++)
    {
        Contact* contact = contacts[i];
        simulation::tree::GNode* group = contactGroup[i];
        while (group!=NULL && mergedGroups.find(group)!=mergedGroups.end())
            group = mergedGroups[group];
        if (group!=NULL)
            contact->createResponse(group);
        else
            contact->createResponse(scene);
    }

    if (node) t0 = node->endTime(t0, "collision/contacts", this);

    // delete removed groups
    for (std::vector<simulation::tree::GNode*>::iterator it = removedGroup.begin(); it!=removedGroup.end(); ++it)
        delete *it;
    removedGroup.clear();

    // finally recreate group vector
    groupVec.clear();
    for (std::set<simulation::tree::GNode*>::iterator it = groupSet.begin(); it!=groupSet.end(); ++it)
        groupVec.push_back(*it);
    //if (!groupVec.empty())
    //	std::cout << groupVec.size()<<" collision groups created."<<std::endl;
}

void DefaultCollisionGroupManager::clearGroups(core::objectmodel::BaseContext* /*scene*/)
{
    for (std::set<simulation::tree::GNode*>::iterator it = groupSet.begin(); it!=groupSet.end(); ++it)
    {
        simulation::tree::GNode* group = *it;
        simulation::tree::GNode* parent = group->parent;
        while(!group->child.empty())
            parent->moveChild(*group->child.begin());
        while(!group->object.empty())
        {
            core::objectmodel::BaseObject* obj = *group->object.begin();
            group->removeObject(obj);
            delete obj;
        }
        parent->removeChild(group);
        delete group;
    }

    groupSet.clear();
    groupVec.clear();
}

simulation::tree::GNode* DefaultCollisionGroupManager::getIntegrationNode(core::CollisionModel* model)
{
    simulation::tree::GNode* node = dynamic_cast<simulation::tree::GNode*>(model->getContext());
    simulation::tree::GNode* lastSolver = NULL;
    while (node!=NULL)
    {
        if (!node->solver.empty()) lastSolver = node;
        node = node->parent;
    }
    return lastSolver;
}

// Sylvere F. : change the name of function, because under Visual C++ it doesn't compile

// Jeremie A. : put the methods inside a namespace instead of a class,
// for g++ 3.4 compatibility

namespace SolverMergers
{

// First the easy cases...

OdeSolver* createSolverEulerEuler(odesolver::EulerSolver& solver1, odesolver::EulerSolver& /*solver2*/)
{
    return new odesolver::EulerSolver(solver1);
}

OdeSolver* createSolverRungeKutta4RungeKutta4(odesolver::RungeKutta4Solver& solver1, odesolver::RungeKutta4Solver& /*solver2*/)
{
    return new odesolver::RungeKutta4Solver(solver1);
}

OdeSolver* createSolverCGImplicitCGImplicit(odesolver::CGImplicitSolver& solver1, odesolver::CGImplicitSolver& solver2)
{
    odesolver::CGImplicitSolver* solver = new odesolver::CGImplicitSolver();
    solver->f_maxIter.setValue( solver1.f_maxIter.getValue() > solver2.f_maxIter.getValue() ? solver1.f_maxIter.getValue() : solver2.f_maxIter.getValue() );

    solver->f_smallDenominatorThreshold.setValue( solver1.f_smallDenominatorThreshold.getValue() < solver2.f_smallDenominatorThreshold.getValue() ? solver1.f_smallDenominatorThreshold.getValue() : solver2.f_smallDenominatorThreshold.getValue());

    solver->f_rayleighStiffness.setValue( solver1.f_rayleighStiffness.getValue() < solver2.f_rayleighStiffness.getValue() ? solver1.f_rayleighStiffness.getValue() : solver2.f_rayleighStiffness.getValue() );

    solver->f_rayleighMass.setValue( solver1.f_rayleighMass.getValue() < solver2.f_rayleighMass.getValue() ? solver1.f_rayleighMass.getValue() : solver2.f_rayleighMass.getValue() );
    return solver;
}

// Then the other, with the policy of taking the more precise solver

OdeSolver* createSolverRungeKutta4Euler(odesolver::RungeKutta4Solver& solver1, odesolver::EulerSolver& /*solver2*/)
{
    return new odesolver::RungeKutta4Solver(solver1);
}

OdeSolver* createSolverCGImplicitEuler(odesolver::CGImplicitSolver& solver1, odesolver::EulerSolver& /*solver2*/)
{
    return new odesolver::CGImplicitSolver(solver1);
}

OdeSolver* createSolverCGImplicitRungeKutta4(odesolver::CGImplicitSolver& solver1, odesolver::RungeKutta4Solver& /*solver2*/)
{
    return new odesolver::CGImplicitSolver(solver1);
}

} // namespace SolverMergers


using namespace SolverMergers;

core::componentmodel::behavior::OdeSolver* SolverMerger::merge(core::componentmodel::behavior::OdeSolver* solver1, core::componentmodel::behavior::OdeSolver* solver2)
{
    static SolverMerger instance;
    return instance.solverDispatcher.go(*solver1, *solver2);
}

SolverMerger::SolverMerger()
{
    solverDispatcher.add<odesolver::EulerSolver,odesolver::EulerSolver,createSolverEulerEuler,false>();
    solverDispatcher.add<odesolver::RungeKutta4Solver,odesolver::RungeKutta4Solver,createSolverRungeKutta4RungeKutta4,false>();
    solverDispatcher.add<odesolver::CGImplicitSolver,odesolver::CGImplicitSolver,createSolverCGImplicitCGImplicit,false>();
    solverDispatcher.add<odesolver::RungeKutta4Solver,odesolver::EulerSolver,createSolverRungeKutta4Euler,true>();
    solverDispatcher.add<odesolver::CGImplicitSolver,odesolver::EulerSolver,createSolverCGImplicitEuler,true>();
    solverDispatcher.add<odesolver::CGImplicitSolver,odesolver::RungeKutta4Solver,createSolverCGImplicitRungeKutta4,true>();
}

}// namespace collision

} // namespace component

} // namespace Sofa
