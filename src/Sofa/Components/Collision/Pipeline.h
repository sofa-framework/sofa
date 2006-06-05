#ifndef SOFA_COMPONENTS_COLLISION_PIPELINE_H
#define SOFA_COMPONENTS_COLLISION_PIPELINE_H

#include "Sofa/Abstract/CollisionModel.h"
#include "Sofa/Abstract/CollisionElement.h"
#include "Intersection.h"
#include "BroadPhaseDetection.h"
#include "NarrowPhaseDetection.h"
#include "DetectionOutput.h"
#include "ContactManager.h"
#include "CollisionGroupManager.h"

#include <vector>

namespace Sofa
{

namespace Components
{

namespace Collision
{

class Pipeline : public virtual Abstract::BaseObject
{
protected:
    std::vector<DetectionOutput*> detectionOutputs;

    std::vector<Intersection*> intersectionMethods;
    std::vector<BroadPhaseDetection*> broadPhaseDetections;
    std::vector<NarrowPhaseDetection*> narrowPhaseDetections;
    std::vector<ContactManager*> contactManagers;
    std::vector<CollisionGroupManager*> groupManagers;

    Intersection* intersectionMethod;
    BroadPhaseDetection* broadPhaseDetection;
    NarrowPhaseDetection* narrowPhaseDetection;
    ContactManager* contactManager;
    CollisionGroupManager* groupManager;

public:
    Pipeline();

    virtual ~Pipeline();

    virtual void init();

    virtual void computeCollisions();

    virtual void startDetection(const std::vector<Abstract::CollisionModel*>& collisionModels) = 0;
    virtual std::vector<DetectionOutput*>& getDetectionOutputs() { return detectionOutputs; }
};

} // namespace Collision

} // namespace Components

} // namespace Sofa

#endif
