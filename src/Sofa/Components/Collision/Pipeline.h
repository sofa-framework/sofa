#ifndef SOFA_COMPONENTS_COLLISION_PIPELINE_H
#define SOFA_COMPONENTS_COLLISION_PIPELINE_H

#include "Sofa/Abstract/CollisionModel.h"
#include "Sofa/Abstract/CollisionElement.h"
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

class Pipeline : public virtual Abstract::Base
{
protected:
    std::vector<DetectionOutput*> detectionOutputs;

    std::vector<BroadPhaseDetection*> broadPhaseDetections;
    std::vector<NarrowPhaseDetection*> narrowPhaseDetections;
    std::vector<ContactManager*> contactManagers;
    std::vector<CollisionGroupManager*> groupManagers;

    BroadPhaseDetection* broadPhaseDetection;
    NarrowPhaseDetection* narrowPhaseDetection;
    ContactManager* contactManager;
    CollisionGroupManager* groupManager;

public:
    Pipeline()
        : broadPhaseDetection(NULL),
          narrowPhaseDetection(NULL),
          contactManager(NULL),
          groupManager(NULL)
    {
    }

    virtual ~Pipeline() { }

    void addBroadPhaseDetection (BroadPhaseDetection *bp) { broadPhaseDetections.push_back(bp); if (broadPhaseDetection==NULL) broadPhaseDetection=bp; }
    void addNarrowPhaseDetection (NarrowPhaseDetection *np) { narrowPhaseDetections.push_back(np); if (narrowPhaseDetection==NULL) narrowPhaseDetection=np; }
    void addContactManager (ContactManager *m) { contactManagers.push_back(m); if (contactManager==NULL) contactManager=m; }
    void addGroupManager (CollisionGroupManager *m) { groupManagers.push_back(m); if (groupManager==NULL) groupManager=m; }

    virtual void startDetection(const std::vector<Abstract::CollisionModel*>& collisionModels) = 0;
    virtual std::vector<DetectionOutput*>& getDetectionOutputs() { return detectionOutputs; }
};

} // namespace Collision

} // namespace Components

} // namespace Sofa

#endif
