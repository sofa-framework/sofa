#ifndef SOFA_COMPONENTS_PIPELINESOFA_H
#define SOFA_COMPONENTS_PIPELINESOFA_H

#include "Collision/Pipeline.h"
#include "Sofa/Abstract/VisualModel.h"

namespace Sofa
{

namespace Components
{

class PipelineSofa : public Collision::Pipeline, public Abstract::VisualModel
{
protected:
    bool verbose_;
    bool draw_;
    int depth_;
public:
    PipelineSofa();

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
    virtual void doCollisionDetection(const std::vector<Abstract::CollisionModel*>& collisionModels);
    /// Add collision response in the simulation graph
    virtual void doCollisionResponse();

public:
    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }
};

} // namespace Components

} // namespace Sofa

#endif
