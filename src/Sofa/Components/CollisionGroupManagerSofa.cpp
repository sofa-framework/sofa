#include "CollisionGroupManagerSofa.h"
#include "Scene.h"
#include "Common/FnDispatcher.h"
#include "Common/FnDispatcher.inl"
#include "Sofa/Abstract/DynamicModel.h"
#include "CollisionGroupManagerSofa.h"
#include "EulerSolver.h"
#include "RungeKutta4Solver.h"
#include "CGImplicitSolver.h"
#include "XML/CollisionGroupNode.h"

namespace Sofa
{

namespace Components
{

using namespace Common;
using namespace Collision;

void create(CollisionGroupManagerSofa*& obj, XML::Node<CollisionGroupManager>* arg)
{
    obj = new CollisionGroupManagerSofa(arg->getName());
}

SOFA_DECL_CLASS(CollisionGroupManagerSofa)

Creator<XML::CollisionGroupNode::Factory, CollisionGroupManagerSofa> CollisionGroupManagerSofaClass("default");


template class FnDispatcher<Core::OdeSolver, Core::OdeSolver*>;
typedef FnDispatcher<Core::OdeSolver, Core::OdeSolver*> SolverDispatcher;

CollisionGroupManagerSofa::CollisionGroupManagerSofa(const std::string& name)
    : name(name)
{
}

CollisionGroupManagerSofa::~CollisionGroupManagerSofa()
{
}

void CollisionGroupManagerSofa::createGroups(Scene* scene, const std::vector<Contact*>& contacts)
{
    // Map storing group merging history
    std::map<Core::MechanicalGroup*, Core::MechanicalGroup*> mergedGroups;
    std::vector<Core::MechanicalGroup*> contactGroup;
    contactGroup.reserve(contacts.size());
    for(std::vector<Contact*>::const_iterator cit = contacts.begin(); cit != contacts.end(); cit++)
    {
        Contact* contact = *cit;
        Abstract::BehaviorModel* object1 = contact->getCollisionModels().first->getObject();
        Abstract::BehaviorModel* object2 = contact->getCollisionModels().second->getObject();
        Core::MechanicalGroup* group = NULL;
        if (object1==NULL && object2==NULL)
        {
            // both objects are static
        }
        else
        {
            // if one object is static, consider this collision as self-collision
            if (object1==NULL) object1 = object2;
            if (object2==NULL) object2 = object1;
            // For now we only support merging mechanical groups
            Core::MechanicalGroup* group1 = dynamic_cast<Core::MechanicalGroup*>(object1);
            Core::MechanicalGroup* group2 = dynamic_cast<Core::MechanicalGroup*>(object2);
            if (group1==NULL || group2==NULL)
            {
                // Unknown groups
            }
            else if (group1 == group2)
            {
                // same group, no new group necessary
                //contact->createResponse(group1);
                group = group1;
            }
            else
            {
                // we need to merge the groups
                // if solvers are compatible...
                OdeSolver* solver = SolverDispatcher::Go(*group1->getSolver(), *group2->getSolver());
                if (solver!=NULL)
                {
                    bool group1IsColl = groupSet.find(group1)!=groupSet.end();
                    bool group2IsColl = groupSet.find(group2)!=groupSet.end();
                    if (!group1IsColl && !group2IsColl)
                    {
                        // create a new group
                        group = new Core::MechanicalGroup();
                        scene->removeBehaviorModel(group1);
                        scene->removeBehaviorModel(group2);
                        group->addObject(group1);
                        group->addObject(group2);
                        scene->addBehaviorModel(group);
                        groupSet.insert(group);
                    }
                    else if (group1IsColl)
                    {
                        group = group1;
                        scene->removeBehaviorModel(group2);
                        // merge group2 in group1
                        if (!group2IsColl)
                        {
                            group->addObject(group2);
                        }
                        else
                        {
                            // merge groups and remove group2
                            const std::vector<Abstract::DynamicModel*>& objects = group2->getObjects();
                            for (std::vector<Abstract::DynamicModel*>::const_iterator ito = objects.begin(); ito != objects.end(); ++ito)
                                group->addObject(*ito);
                            const std::vector<Core::InteractionForceField*>& forcefields = group2->getForceFields();
                            for (std::vector<Core::InteractionForceField*>::const_iterator itf = forcefields.begin(); itf != forcefields.end(); ++itf)
                                group->addForceField(*itf);
                            groupSet.erase(group2);
                            group2->setSolver(NULL);
                            delete group2;
                            mergedGroups[group2] = group;
                        }
                    }
                    else
                    {
                        // group1 is not a collision group while group2 is
                        group = group2;
                        scene->removeBehaviorModel(group1);
                        group->addObject(group1);
                    }
                    group->setSolver(solver);
                    solver->setGroup(group);
                    //contact->createResponse(group);
                }
            }
        }
        contactGroup.push_back(group);
    }

    // now that the groups are final, attach contacts' response
    for(unsigned int i=0; i<contacts.size(); i++)
    {
        Contact* contact = contacts[i];
        Core::MechanicalGroup* group = contactGroup[i];
        while (group!=NULL && mergedGroups.find(group)!=mergedGroups.end())
            group = mergedGroups[group];
        if (group!=NULL)
            contact->createResponse(group);
        else
            contact->createResponse(scene);
    }

    // finally recreate group vector
    groupVec.clear();
    for (std::set<Core::MechanicalGroup*>::iterator it = groupSet.begin(); it!=groupSet.end(); ++it)
        groupVec.push_back(*it);
    if (!groupVec.empty())
        std::cout << groupVec.size()<<" collision groups created."<<std::endl;
}

void CollisionGroupManagerSofa::clearGroups(Scene* scene)
{
    for (std::set<Core::MechanicalGroup*>::iterator it = groupSet.begin(); it!=groupSet.end(); ++it)
    {
        Core::MechanicalGroup* group = *it;
        const std::vector<Abstract::DynamicModel*>& objects = group->getObjects();
        for (std::vector<Abstract::DynamicModel*>::const_iterator it2 = objects.begin(); it2 != objects.end(); ++it2)
        {
            Abstract::DynamicModel* object = *it2;
            object->setObject(NULL);
            if (dynamic_cast<Abstract::BehaviorModel*>(object)!=NULL)
                scene->addBehaviorModel(dynamic_cast<Abstract::BehaviorModel*>(object));
            else
                scene->addDynamicModel(object);
        }
        scene->removeBehaviorModel(group);
        group->setSolver(NULL);
        delete group;
    }
    groupSet.clear();
    groupVec.clear();
}

class SolverMerger
{
public:

// First the easy cases...

// Sylvere F. : change the name of function, because under Visual C++ it doesn't compile
// createSolver1, 2, etc.

    static OdeSolver* createSolver1(EulerSolver& solver1, EulerSolver& /*solver2*/)
    {
        return new EulerSolver(solver1);
    }

    static OdeSolver* createSolver2(RungeKutta4Solver& solver1, RungeKutta4Solver& /*solver2*/)
    {
        return new RungeKutta4Solver(solver1);
    }

    static OdeSolver* createSolver3(CGImplicitSolver& solver1, CGImplicitSolver& solver2)
    {
        CGImplicitSolver* solver = new CGImplicitSolver();
        solver->maxCGIter = solver1.maxCGIter > solver2.maxCGIter ? solver1.maxCGIter : solver2.maxCGIter;
        solver->smallDenominatorThreshold = solver1.smallDenominatorThreshold < solver2.smallDenominatorThreshold ? solver1.smallDenominatorThreshold : solver2.smallDenominatorThreshold;
        solver->rayleighStiffness = solver1.rayleighStiffness < solver2.rayleighStiffness ? solver1.rayleighStiffness : solver2.rayleighStiffness;
        return solver;
    }

// Then the other, with the policy of taking the more precise solver

    static OdeSolver* createSolver4(RungeKutta4Solver& solver1, EulerSolver& /*solver2*/)
    {
        return new RungeKutta4Solver(solver1);
    }

    static OdeSolver* createSolver5(CGImplicitSolver& solver1, EulerSolver& /*solver2*/)
    {
        return new CGImplicitSolver(solver1);
    }

    static OdeSolver* createSolver6(CGImplicitSolver& solver1, RungeKutta4Solver& /*solver2*/)
    {
        return new CGImplicitSolver(solver1);
    }

protected:

    SolverMerger ()
    {
        SolverDispatcher::Add<EulerSolver,EulerSolver,createSolver1,false>();
        SolverDispatcher::Add<RungeKutta4Solver,RungeKutta4Solver,createSolver2,false>();
        SolverDispatcher::Add<CGImplicitSolver,CGImplicitSolver,createSolver3,false>();
        SolverDispatcher::Add<RungeKutta4Solver,EulerSolver,createSolver4,true>();
        SolverDispatcher::Add<CGImplicitSolver,EulerSolver,createSolver5,true>();
        SolverDispatcher::Add<CGImplicitSolver,RungeKutta4Solver,createSolver6,true>();
    }
    static SolverMerger instance;
};

SolverMerger SolverMerger::instance;

} // namespace Components

} // namespace Sofa
