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

    virtual void reset();

    /// Remove collision response from last step
    virtual void computeCollisionReset();
    /// Detect new collisions. Note that this step must not modify the simulation graph
    virtual void computeCollisionDetection();
    /// Add collision response in the simulation graph
    virtual void computeCollisionResponse();

    void computeCollisions()
    {
        computeCollisionReset();
        computeCollisionDetection();
        computeCollisionResponse();
    }

    std::vector<DetectionOutput*>& getDetectionOutputs() { return detectionOutputs; }

protected:
    /// Remove collision response from last step
    virtual void doCollisionReset() = 0;
    /// Detect new collisions. Note that this step must not modify the simulation graph
    virtual void doCollisionDetection(const std::vector<Abstract::CollisionModel*>& collisionModels) = 0;
    /// Add collision response in the simulation graph
    virtual void doCollisionResponse() = 0;
};

} // namespace Collision

} // namespace Components

} // namespace Sofa

#endif
