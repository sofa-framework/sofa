#ifndef SOFA_COMPONENT_COLLISION_BRUTEFORCEDETECTION_H
#define SOFA_COMPONENT_COLLISION_BRUTEFORCEDETECTION_H

#include <sofa/core/componentmodel/collision/BroadPhaseDetection.h>
#include <sofa/core/componentmodel/collision/NarrowPhaseDetection.h>
#include <sofa/core/CollisionElement.h>
#include <sofa/core/VisualModel.h>
#include <sofa/defaulttype/Vec.h>
#include <set>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

// inherit of VisualModel for debugging, then we can see the voxel grid
class BruteForceDetection :
    public core::componentmodel::collision::BroadPhaseDetection,
    public core::componentmodel::collision::NarrowPhaseDetection,
    public core::VisualModel
{
private:
    std::vector<core::CollisionModel*> collisionModels;
    DataField<bool> bDraw;

public:

    BruteForceDetection();

    void setDraw(bool val) { bDraw.setValue(val); }

    void addCollisionModel (core::CollisionModel *cm);
    void addCollisionPair (const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair);

    virtual void clearBroadPhase()
    {
        core::componentmodel::collision::BroadPhaseDetection::clearBroadPhase();
        collisionModels.clear();
    }

    /* for debugging, VisualModel */
    void draw();
    void initTextures() { }
    void update() { }
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
