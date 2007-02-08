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
protected:
    bool verbose_;
    bool draw_;
    int depth_;
public:
    DefaultPipeline();

    void setVerbose(bool v) { verbose_ = v;    }
    bool getVerbose() const { return verbose_; }
    void setDraw(bool v)    { draw_ = v;       }
    bool getDraw() const    { return draw_;    }
    void setDepth(int v)    { depth_ = v;      }
    int getDepth() const    { return depth_;   }

    virtual const char* getTypeName() const { return "CollisionPipeline"; }

protected:
    // -- Pipeline interface

    /// Remove collision response from last step
    virtual void doCollisionReset();
    /// Detect new collisions. Note that this step must not modify the simulation graph
    virtual void doCollisionDetection(const std::vector<core::CollisionModel*>& collisionModels);
    /// Add collision response in the simulation graph
    virtual void doCollisionResponse();

public:
    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
