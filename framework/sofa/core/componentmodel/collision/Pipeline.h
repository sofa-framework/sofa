#ifndef SOFA_CORE_COMPONENTMODEL_COLLISION_PIPELINE_H
#define SOFA_CORE_COMPONENTMODEL_COLLISION_PIPELINE_H

#include <sofa/core/CollisionModel.h>
#include <sofa/core/CollisionElement.h>
#include <sofa/core/componentmodel/collision/Intersection.h>
#include <sofa/core/componentmodel/collision/BroadPhaseDetection.h>
#include <sofa/core/componentmodel/collision/NarrowPhaseDetection.h>
#include <sofa/core/componentmodel/collision/DetectionOutput.h>
#include <sofa/core/componentmodel/collision/ContactManager.h>
#include <sofa/core/componentmodel/collision/CollisionGroupManager.h>

#include <vector>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace collision
{

class Pipeline : public virtual sofa::core::objectmodel::BaseObject
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
    virtual void doCollisionDetection(const std::vector<core::CollisionModel*>& collisionModels) = 0;
    /// Add collision response in the simulation graph
    virtual void doCollisionResponse() = 0;
};

} // namespace collision

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
