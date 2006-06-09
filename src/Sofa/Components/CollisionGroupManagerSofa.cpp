#include "CollisionGroupManagerSofa.h"
#include "Sofa/Abstract/CollisionModel.h"
#include "Common/config.h"
#include "Common/FnDispatcher.h"
#include "Common/FnDispatcher.inl"
#include "CollisionGroupManagerSofa.h"
#include "EulerSolver.h"
#include "RungeKutta4Solver.h"
#include "CGImplicitSolver.h"
#include "Common/ObjectFactory.h"

#include <string.h>

namespace Sofa
{

namespace Components
{

using namespace Common;
using namespace Core;
using namespace Collision;
using namespace Graph;

void create(CollisionGroupManagerSofa*& obj, ObjectDescription* arg)
{
    obj = new CollisionGroupManagerSofa;
}

SOFA_DECL_CLASS(CollisionGroupManagerSofa)

Creator<ObjectFactory, CollisionGroupManagerSofa> CollisionGroupManagerSofaClass("CollisionGroup");

class SolverMerger
{
public:
    static Core::OdeSolver* merge(Core::OdeSolver* solver1, Core::OdeSolver* solver2);

protected:

    FnDispatcher<Core::OdeSolver, Core::OdeSolver*> solverDispatcher;

    SolverMerger ();
};

CollisionGroupManagerSofa::CollisionGroupManagerSofa()
{
}

CollisionGroupManagerSofa::~CollisionGroupManagerSofa()
{
}

void CollisionGroupManagerSofa::createGroups(Abstract::BaseContext* scene, const std::vector<Contact*>& contacts)
{
    int groupIndex = 1;
    GNode* groot = dynamic_cast<GNode*>(scene);
    if (groot==NULL)
    {
        std::cerr << "CollisionGroupManagerSofa only support graph-based scenes.\n";
        return;
    }
    // Map storing group merging history
    std::map<GNode*, GNode*> mergedGroups;
    std::vector<GNode*> contactGroup;
    contactGroup.reserve(contacts.size());
    for(std::vector<Contact*>::const_iterator cit = contacts.begin(); cit != contacts.end(); cit++)
    {
        Contact* contact = *cit;
        GNode* group1 = getIntegrationNode(contact->getCollisionModels().first);
        GNode* group2 = getIntegrationNode(contact->getCollisionModels().second);
        GNode* group = NULL;
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
                GNode* parent = group1->parent;
                bool group1IsColl = groupSet.find(group1)!=groupSet.end();
                bool group2IsColl = groupSet.find(group2)!=groupSet.end();
                if (!group1IsColl && !group2IsColl)
                {
                    char groupName[32];
                    snprintf(groupName,sizeof(groupName),"collision%d",groupIndex++);
                    // create a new group
                    group = new GNode(groupName);
                    parent->addChild(group);
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
                        delete group2;
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

    // now that the groups are final, attach contacts' response
    for(unsigned int i=0; i<contacts.size(); i++)
    {
        Contact* contact = contacts[i];
        GNode* group = contactGroup[i];
        while (group!=NULL && mergedGroups.find(group)!=mergedGroups.end())
            group = mergedGroups[group];
        if (group!=NULL)
            contact->createResponse(group);
        else
            contact->createResponse(scene);
    }

    // finally recreate group vector
    groupVec.clear();
    for (std::set<GNode*>::iterator it = groupSet.begin(); it!=groupSet.end(); ++it)
        groupVec.push_back(*it);
    //if (!groupVec.empty())
    //	std::cout << groupVec.size()<<" collision groups created."<<std::endl;
}

void CollisionGroupManagerSofa::clearGroups(Abstract::BaseContext* /*scene*/)
{
    for (std::set<GNode*>::iterator it = groupSet.begin(); it!=groupSet.end(); ++it)
    {
        GNode* group = *it;
        GNode* parent = group->parent;
        while(!group->child.empty())
            parent->moveChild(*group->child.begin());
        while(!group->object.empty())
        {
            Abstract::BaseObject* obj = *group->object.begin();
            group->removeObject(obj);
            delete obj;
        }
        parent->removeChild(group);
        delete group;
    }

    groupSet.clear();
    groupVec.clear();
}

Graph::GNode* CollisionGroupManagerSofa::getIntegrationNode(Abstract::CollisionModel* model)
{
    GNode* node = dynamic_cast<GNode*>(model->getContext());
    GNode* lastSolver = NULL;
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

OdeSolver* createSolverEulerEuler(EulerSolver& solver1, EulerSolver& /*solver2*/)
{
    return new EulerSolver(solver1);
}

OdeSolver* createSolverRungeKutta4RungeKutta4(RungeKutta4Solver& solver1, RungeKutta4Solver& /*solver2*/)
{
    return new RungeKutta4Solver(solver1);
}

OdeSolver* createSolverCGImplicitCGImplicit(CGImplicitSolver& solver1, CGImplicitSolver& solver2)
{
    CGImplicitSolver* solver = new CGImplicitSolver();
    solver->maxCGIter = solver1.maxCGIter > solver2.maxCGIter ? solver1.maxCGIter : solver2.maxCGIter;
    solver->smallDenominatorThreshold = solver1.smallDenominatorThreshold < solver2.smallDenominatorThreshold ? solver1.smallDenominatorThreshold : solver2.smallDenominatorThreshold;
    solver->rayleighStiffness = solver1.rayleighStiffness < solver2.rayleighStiffness ? solver1.rayleighStiffness : solver2.rayleighStiffness;
    return solver;
}

// Then the other, with the policy of taking the more precise solver

OdeSolver* createSolverRungeKutta4Euler(RungeKutta4Solver& solver1, EulerSolver& /*solver2*/)
{
    return new RungeKutta4Solver(solver1);
}

OdeSolver* createSolverCGImplicitEuler(CGImplicitSolver& solver1, EulerSolver& /*solver2*/)
{
    return new CGImplicitSolver(solver1);
}

OdeSolver* createSolverCGImplicitRungeKutta4(CGImplicitSolver& solver1, RungeKutta4Solver& /*solver2*/)
{
    return new CGImplicitSolver(solver1);
}

} // namespace SolverMergers

using namespace SolverMergers;

Core::OdeSolver* SolverMerger::merge(Core::OdeSolver* solver1, Core::OdeSolver* solver2)
{
    static SolverMerger instance;
    return instance.solverDispatcher.go(*solver1, *solver2);
}

SolverMerger::SolverMerger()
{
    solverDispatcher.add<EulerSolver,EulerSolver,createSolverEulerEuler,false>();
    solverDispatcher.add<RungeKutta4Solver,RungeKutta4Solver,createSolverRungeKutta4RungeKutta4,false>();
    solverDispatcher.add<CGImplicitSolver,CGImplicitSolver,createSolverCGImplicitCGImplicit,false>();
    solverDispatcher.add<RungeKutta4Solver,EulerSolver,createSolverRungeKutta4Euler,true>();
    solverDispatcher.add<CGImplicitSolver,EulerSolver,createSolverCGImplicitEuler,true>();
    solverDispatcher.add<CGImplicitSolver,RungeKutta4Solver,createSolverCGImplicitRungeKutta4,true>();
}

} // namespace Components

} // namespace Sofa
