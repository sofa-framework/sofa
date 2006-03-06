#ifndef SOFA_COMPONENTS_PIPELINESOFA_H
#define SOFA_COMPONENTS_PIPELINESOFA_H

#include "Collision/Pipeline.h"

namespace Sofa
{

namespace Components
{

class PipelineSofa : public Collision::Pipeline
{
protected:
    bool verbose;
public:
    PipelineSofa(bool verbose);
    virtual void startDetection(const std::vector<Abstract::CollisionModel*>& collisionModels);
};

} // namespace Components

} // namespace Sofa

#endif
