#include <sofa/core/componentmodel/collision/Pipeline.h>
#include <sofa/component/collision/DiscreteIntersection.h>
#include <sofa/simulation/tree/GNode.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace collision
{
using namespace core::objectmodel;
using namespace core::componentmodel::behavior;

Pipeline::Pipeline()
    : intersectionMethod(NULL),
      broadPhaseDetection(NULL),
      narrowPhaseDetection(NULL),
      contactManager(NULL),
      groupManager(NULL)
{
}

Pipeline::~Pipeline()
{
}

void Pipeline::init()
{
    simulation::tree::GNode* root = dynamic_cast<simulation::tree::GNode*>(getContext());
    if(root == NULL) return;
    intersectionMethods.clear();
    root->getTreeObjects<Intersection>(&intersectionMethods);
    intersectionMethod = (intersectionMethods.empty() ? NULL : intersectionMethods[0]);
    broadPhaseDetections.clear();
    root->getTreeObjects<BroadPhaseDetection>(&broadPhaseDetections);
    broadPhaseDetection = (broadPhaseDetections.empty() ? NULL : broadPhaseDetections[0]);
    narrowPhaseDetections.clear();
    root->getTreeObjects<NarrowPhaseDetection>(&narrowPhaseDetections);
    narrowPhaseDetection = (narrowPhaseDetections.empty() ? NULL : narrowPhaseDetections[0]);
    contactManagers.clear();
    root->getTreeObjects<ContactManager>(&contactManagers);
    contactManager = (contactManagers.empty() ? NULL : contactManagers[0]);
    groupManagers.clear();
    root->getTreeObjects<CollisionGroupManager>(&groupManagers);
    groupManager = (groupManagers.empty() ? NULL : groupManagers[0]);

    if (intersectionMethod==NULL)
        intersectionMethod = new sofa::component::collision::DiscreteIntersection;
}

void Pipeline::reset()
{
    computeCollisionReset();
}

void Pipeline::computeCollisionReset()
{
    simulation::tree::GNode* root = dynamic_cast<simulation::tree::GNode*>(getContext());
    if(root == NULL) return;
    if (broadPhaseDetection!=NULL && broadPhaseDetection->getIntersectionMethod()!=intersectionMethod)
        broadPhaseDetection->setIntersectionMethod(intersectionMethod);
    if (narrowPhaseDetection!=NULL && narrowPhaseDetection->getIntersectionMethod()!=intersectionMethod)
        narrowPhaseDetection->setIntersectionMethod(intersectionMethod);
    if (contactManager!=NULL && contactManager->getIntersectionMethod()!=intersectionMethod)
        contactManager->setIntersectionMethod(intersectionMethod);
    doCollisionReset();
}

void Pipeline::computeCollisionDetection()
{
    simulation::tree::GNode* root = dynamic_cast<simulation::tree::GNode*>(getContext());
    if(root == NULL) return;
    std::vector<CollisionModel*> collisionModels;
    root->getTreeObjects<CollisionModel>(&collisionModels);
    doCollisionDetection(collisionModels);
}

void Pipeline::computeCollisionResponse()
{
    simulation::tree::GNode* root = dynamic_cast<simulation::tree::GNode*>(getContext());
    if(root == NULL) return;
    doCollisionResponse();
}

} // namespace collision

} // namespace componentmodel

} // namespace core

} // namespace sofa

