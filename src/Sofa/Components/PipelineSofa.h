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

    // -- Pipeline interface
    virtual void startDetection(const std::vector<Abstract::CollisionModel*>& collisionModels);

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }
};

} // namespace Components

} // namespace Sofa

#endif
