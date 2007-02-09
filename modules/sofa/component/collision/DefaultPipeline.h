#ifndef SOFA_COMPONENT_COLLISION_DEFAULTPIPELINE_H
#define SOFA_COMPONENT_COLLISION_DEFAULTPIPELINE_H

#include <sofa/core/componentmodel/collision/Pipeline.h>
#include <sofa/core/VisualModel.h>

namespace sofa
{

namespace component
{

namespace collision
{

class DefaultPipeline : public core::componentmodel::collision::Pipeline, public core::VisualModel
{
public:
    DataField<bool> bVerbose;
    DataField<bool> bDraw;
    DataField<int> depth;

    DefaultPipeline();

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }

protected:
    // -- Pipeline interface

    /// Remove collision response from last step
    virtual void doCollisionReset();
    /// Detect new collisions. Note that this step must not modify the simulation graph
    virtual void doCollisionDetection(const std::vector<core::CollisionModel*>& collisionModels);
    /// Add collision response in the simulation graph
    virtual void doCollisionResponse();
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
