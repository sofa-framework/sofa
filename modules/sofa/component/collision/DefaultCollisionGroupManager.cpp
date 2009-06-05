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
#include <sofa/component/collision/DefaultCollisionGroupManager.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/CollisionModel.h>
// #include <sofa/helper/system/config.h>
#include <sofa/helper/FnDispatcher.h>
#include <sofa/helper/FnDispatcher.inl>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/component/odesolver/RungeKutta4Solver.h>
#include <sofa/component/odesolver/CGImplicitSolver.h>
#include <sofa/component/odesolver/StaticSolver.h>
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>
// #include <string.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/simulation/common/Simulation.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace core::componentmodel::behavior;
using namespace core::componentmodel::collision;

SOFA_DECL_CLASS(DefaultCollisionGroupManager);

int DefaultCollisionGroupManagerClass = core::RegisterObject("Responsible for gathering colliding objects in the same group, for consistent time integration")
        .add< DefaultCollisionGroupManager >()
        .addAlias("CollisionGroup")
        ;

typedef std::pair<OdeSolver*,LinearSolver*> SolverSet;

class SolverMerger
{
public:
    static SolverSet merge(core::componentmodel::behavior::OdeSolver* solver1, core::componentmodel::behavior::OdeSolver* solver2);

protected:

    FnDispatcher<core::componentmodel::behavior::OdeSolver, SolverSet> solverDispatcher;

    SolverMerger ();
};

DefaultCollisionGroupManager::DefaultCollisionGroupManager()
{
}

DefaultCollisionGroupManager::~DefaultCollisionGroupManager()
{
}

simulation::Node* DefaultCollisionGroupManager::buildCollisionGroup()
{
    return simulation::getSimulation()->newNode("CollisionGroup");
}

void DefaultCollisionGroupManager::createGroups(core::objectmodel::BaseContext* scene, const sofa::helper::vector<Contact*>& contacts)
{
    int groupIndex = 1;
    simulation::Node* node = dynamic_cast<simulation::Node*>(scene);
    if (node==NULL)
    {
        serr << "DefaultCollisionGroupManager only support graph-based scenes."<<sendl;
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
        simulation::tree::GNode* group1 = static_cast<simulation::tree::GNode*>(getIntegrationNode(contact->getCollisionModels().first));
        simulation::tree::GNode* group2 = static_cast<simulation::tree::GNode*>(getIntegrationNode(contact->getCollisionModels().second));
        simulation::Node* group = NULL;
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
            SolverSet solver = SolverMerger::merge(group1->solver[0], group2->solver[0]);
            if (solver.first!=NULL)
            {
                simulation::tree::GNode* parent = group1->parent;
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
                    OdeSolver* solver2 = group->solver[0];
                    group->removeObject(solver2);
                    delete solver2;
                }
                if (!group->linearSolver.empty())
                {
                    LinearSolver* solver2 = group->linearSolver[0];
                    group->removeObject(solver2);
                    delete solver2;
                }
                group->addObject(solver.first);
                if (solver.second)
                    group->addObject(solver.second);
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

void DefaultCollisionGroupManager::clearGroups(core::objectmodel::BaseContext* /*scene*/)
{
    for (std::set<simulation::Node*>::iterator it = groupSet.begin(); it!=groupSet.end(); ++it)
    {
        simulation::tree::GNode* group = dynamic_cast<simulation::tree::GNode*>(*it);
        if (group)
        {
            simulation::tree::GNode* parent = group->parent;
            while(!group->child.empty())
                parent->moveChild(*group->child.begin());
            while(!group->object.empty())
            {
                core::objectmodel::BaseObject* obj = *group->object.begin();
                group->removeObject(obj);
                delete obj;
            }
            parent->removeChild((simulation::Node*)group);
            delete group;
        }
    }

    groupSet.clear();
    groups.clear();
}

simulation::Node* DefaultCollisionGroupManager::getIntegrationNode(core::CollisionModel* model)
{
    simulation::Node* node = static_cast<simulation::Node*>(model->getContext());
    helper::vector< core::componentmodel::behavior::OdeSolver *> listSolver;
    node->get< core::componentmodel::behavior::OdeSolver >(&listSolver);

    if (!listSolver.empty()) return static_cast<simulation::Node*>(listSolver.back()->getContext());
    else                     return NULL;
}

// Sylvere F. : change the name of function, because under Visual C++ it doesn't compile

// Jeremie A. : put the methods inside a namespace instead of a class,
// for g++ 3.4 compatibility

namespace SolverMergers
{

// First the easy cases...

SolverSet createSolverEulerEuler(odesolver::EulerSolver& solver1, odesolver::EulerSolver& /*solver2*/)
{
    return SolverSet(new odesolver::EulerSolver(solver1), NULL);
}

SolverSet createSolverRungeKutta4RungeKutta4(odesolver::RungeKutta4Solver& solver1, odesolver::RungeKutta4Solver& /*solver2*/)
{
    return SolverSet(new odesolver::RungeKutta4Solver(solver1), NULL);
}

SolverSet createSolverCGImplicitCGImplicit(odesolver::CGImplicitSolver& solver1, odesolver::CGImplicitSolver& solver2)
{
    odesolver::CGImplicitSolver* solver = new odesolver::CGImplicitSolver;
    solver->f_maxIter.setValue( solver1.f_maxIter.getValue() > solver2.f_maxIter.getValue() ? solver1.f_maxIter.getValue() : solver2.f_maxIter.getValue() );
    solver->f_tolerance.setValue( solver1.f_tolerance.getValue() < solver2.f_tolerance.getValue() ? solver1.f_tolerance.getValue() : solver2.f_tolerance.getValue());
    solver->f_smallDenominatorThreshold.setValue( solver1.f_smallDenominatorThreshold.getValue() < solver2.f_smallDenominatorThreshold.getValue() ? solver1.f_smallDenominatorThreshold.getValue() : solver2.f_smallDenominatorThreshold.getValue());

    solver->f_rayleighStiffness.setValue( solver1.f_rayleighStiffness.getValue() < solver2.f_rayleighStiffness.getValue() ? solver1.f_rayleighStiffness.getValue() : solver2.f_rayleighStiffness.getValue() );

    solver->f_rayleighMass.setValue( solver1.f_rayleighMass.getValue() < solver2.f_rayleighMass.getValue() ? solver1.f_rayleighMass.getValue() : solver2.f_rayleighMass.getValue() );
    solver->f_velocityDamping.setValue( solver1.f_velocityDamping.getValue() > solver2.f_velocityDamping.getValue() ? solver1.f_velocityDamping.getValue() : solver2.f_velocityDamping.getValue());
    return SolverSet(solver, NULL);
}

typedef linearsolver::CGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector> DefaultCGLinearSolver;

LinearSolver* createLinearSolver(OdeSolver* solver1, OdeSolver* solver2)
{
    DefaultCGLinearSolver* lsolver = new DefaultCGLinearSolver;
    DefaultCGLinearSolver* lsolver1 = NULL; if (solver1!=NULL) solver1->getContext()->get(lsolver1, core::objectmodel::BaseContext::SearchDown);
    DefaultCGLinearSolver* lsolver2 = NULL; if (solver2!=NULL) solver2->getContext()->get(lsolver2, core::objectmodel::BaseContext::SearchDown);
    unsigned int maxIter = 0;
    double tolerance = 1.0e10;
    double smallDenominatorThreshold = 1.0e10;
    if (lsolver1)
    {
        if (lsolver1->f_maxIter.getValue() > maxIter) maxIter = lsolver1->f_maxIter.getValue();
        if (lsolver1->f_tolerance.getValue() < tolerance) tolerance = lsolver1->f_tolerance.getValue();
        if (lsolver1->f_smallDenominatorThreshold.getValue() > smallDenominatorThreshold) smallDenominatorThreshold = lsolver1->f_smallDenominatorThreshold.getValue();
    }
    if (lsolver2)
    {
        if (lsolver2->f_maxIter.getValue() > maxIter) maxIter = lsolver2->f_maxIter.getValue();
        if (lsolver2->f_tolerance.getValue() < tolerance) tolerance = lsolver2->f_tolerance.getValue();
        if (lsolver2->f_smallDenominatorThreshold.getValue() > smallDenominatorThreshold) smallDenominatorThreshold = lsolver2->f_smallDenominatorThreshold.getValue();
    }
    if (maxIter > 0) lsolver->f_maxIter.setValue( maxIter );
    if (tolerance < 1.0e10) lsolver->f_tolerance.setValue( tolerance );
    if (smallDenominatorThreshold < 1.0e10) lsolver->f_smallDenominatorThreshold.setValue( smallDenominatorThreshold );
    return lsolver;
}

SolverSet createSolverEulerImplicitEulerImplicit(odesolver::EulerImplicitSolver& solver1, odesolver::EulerImplicitSolver& solver2)
{
    odesolver::EulerImplicitSolver* solver = new odesolver::EulerImplicitSolver;
    solver->f_rayleighStiffness.setValue( solver1.f_rayleighStiffness.getValue() < solver2.f_rayleighStiffness.getValue() ? solver1.f_rayleighStiffness.getValue() : solver2.f_rayleighStiffness.getValue() );

    solver->f_rayleighMass.setValue( solver1.f_rayleighMass.getValue() < solver2.f_rayleighMass.getValue() ? solver1.f_rayleighMass.getValue() : solver2.f_rayleighMass.getValue() );
    solver->f_velocityDamping.setValue( solver1.f_velocityDamping.getValue() > solver2.f_velocityDamping.getValue() ? solver1.f_velocityDamping.getValue() : solver2.f_velocityDamping.getValue());
    return SolverSet(solver, createLinearSolver(&solver1, &solver2));
}

SolverSet createSolverStaticSolver(odesolver::StaticSolver& solver1, odesolver::StaticSolver& solver2)
{
    return SolverSet(new odesolver::StaticSolver(solver1), createLinearSolver(&solver1, &solver2));
}

// Then the other, with the policy of taking the more precise solver

SolverSet createSolverRungeKutta4Euler(odesolver::RungeKutta4Solver& solver1, odesolver::EulerSolver& /*solver2*/)
{
    return SolverSet(new odesolver::RungeKutta4Solver(solver1), NULL);
}

SolverSet createSolverCGImplicitEuler(odesolver::CGImplicitSolver& solver1, odesolver::EulerSolver& /*solver2*/)
{
    return SolverSet(new odesolver::CGImplicitSolver(solver1), NULL);
}

SolverSet createSolverCGImplicitRungeKutta4(odesolver::CGImplicitSolver& solver1, odesolver::RungeKutta4Solver& /*solver2*/)
{
    return SolverSet(new odesolver::CGImplicitSolver(solver1), NULL);
}

SolverSet createSolverEulerImplicitEuler(odesolver::EulerImplicitSolver& solver1, odesolver::EulerSolver& /*solver2*/)
{
    return SolverSet(new odesolver::EulerImplicitSolver(solver1), createLinearSolver(&solver1, NULL));
}

SolverSet createSolverEulerImplicitRungeKutta4(odesolver::EulerImplicitSolver& solver1, odesolver::RungeKutta4Solver& /*solver2*/)
{
    return SolverSet(new odesolver::EulerImplicitSolver(solver1), createLinearSolver(&solver1, NULL));
}

SolverSet createSolverEulerImplicitCGImplicit(odesolver::EulerImplicitSolver& solver1, odesolver::CGImplicitSolver& /*solver2*/)
{
    return SolverSet(new odesolver::EulerImplicitSolver(solver1), createLinearSolver(&solver1, NULL));
}

} // namespace SolverMergers


using namespace SolverMergers;

SolverSet SolverMerger::merge(core::componentmodel::behavior::OdeSolver* solver1, core::componentmodel::behavior::OdeSolver* solver2)
{
    static SolverMerger instance;
    SolverSet obj=instance.solverDispatcher.go(*solver1, *solver2);
#ifdef SOFA_HAVE_EIGEN2
    obj.first->constraintAcc.setValue( (solver1->constraintAcc.getValue() || solver2->constraintAcc.getValue() ) );
    obj.first->constraintVel.setValue( (solver1->constraintVel.getValue() || solver2->constraintVel.getValue() ) );
    obj.first->constraintPos.setValue( (solver1->constraintPos.getValue() || solver2->constraintPos.getValue() ) );
    obj.first->constraintResolution.setValue( (solver1->constraintResolution.getValue() && solver2->constraintResolution.getValue() ) );
    obj.first->numIterations.setValue( std::max(solver1->numIterations.getValue(), solver2->numIterations.getValue() ) );
    obj.first->maxError.setValue( std::min(solver1->maxError.getValue(), solver2->maxError.getValue() ) );
#endif
    return obj;
}

SolverMerger::SolverMerger()
{
    solverDispatcher.add<odesolver::EulerSolver,odesolver::EulerSolver,createSolverEulerEuler,false>();
    solverDispatcher.add<odesolver::RungeKutta4Solver,odesolver::RungeKutta4Solver,createSolverRungeKutta4RungeKutta4,false>();
    solverDispatcher.add<odesolver::CGImplicitSolver,odesolver::CGImplicitSolver,createSolverCGImplicitCGImplicit,false>();
    solverDispatcher.add<odesolver::EulerImplicitSolver,odesolver::EulerImplicitSolver,createSolverEulerImplicitEulerImplicit,false>();
    solverDispatcher.add<odesolver::RungeKutta4Solver,odesolver::EulerSolver,createSolverRungeKutta4Euler,true>();
    solverDispatcher.add<odesolver::CGImplicitSolver,odesolver::EulerSolver,createSolverCGImplicitEuler,true>();
    solverDispatcher.add<odesolver::CGImplicitSolver,odesolver::RungeKutta4Solver,createSolverCGImplicitRungeKutta4,true>();
    solverDispatcher.add<odesolver::EulerImplicitSolver,odesolver::EulerSolver,createSolverEulerImplicitEuler,true>();
    solverDispatcher.add<odesolver::EulerImplicitSolver,odesolver::RungeKutta4Solver,createSolverEulerImplicitRungeKutta4,true>();
    solverDispatcher.add<odesolver::EulerImplicitSolver,odesolver::CGImplicitSolver,createSolverEulerImplicitCGImplicit,true>();
    solverDispatcher.add<odesolver::StaticSolver,odesolver::StaticSolver,createSolverStaticSolver,true>();
}

}// namespace collision

} // namespace component

} // namespace Sofa
