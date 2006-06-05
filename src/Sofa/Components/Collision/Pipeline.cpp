#include "Pipeline.h"
#include "DiscreteIntersection.h"
#include "../Graph/GNode.h"

namespace Sofa
{

namespace Components
{

namespace Collision
{

using namespace Abstract;
using namespace Core;

Pipeline::Pipeline()
    : broadPhaseDetection(NULL),
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
    Graph::GNode* root = dynamic_cast<Graph::GNode*>(getContext());
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
        intersectionMethod = new DiscreteIntersection;
}

void Pipeline::computeCollisions()
{
    Graph::GNode* root = dynamic_cast<Graph::GNode*>(getContext());
    if(root == NULL) return;
    std::vector<CollisionModel*> collisionModels;
    root->getTreeObjects<CollisionModel>(&collisionModels);
    if (broadPhaseDetection!=NULL && broadPhaseDetection->getIntersectionMethod()!=intersectionMethod)
        broadPhaseDetection->setIntersectionMethod(intersectionMethod);
    if (narrowPhaseDetection!=NULL && narrowPhaseDetection->getIntersectionMethod()!=intersectionMethod)
        narrowPhaseDetection->setIntersectionMethod(intersectionMethod);
    startDetection(collisionModels);
}

} // namespace Collision

} // namespace Components

} // namespace Sofa
