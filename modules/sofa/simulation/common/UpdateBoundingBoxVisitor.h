#ifndef SOFA_SIMULATION_COMMON_COMPUTEBOUNDINGBOXVISITOR_H
#define SOFA_SIMULATION_COMMON_COMPUTEBOUNDINGBOXVISITOR_H

#include <sofa/simulation/common/Visitor.h>
#include <sofa/core/ExecParams.h>
namespace sofa
{
namespace simulation
{

class SOFA_SIMULATION_COMMON_API UpdateBoundingBoxVisitor : public Visitor
{
public:

    UpdateBoundingBoxVisitor(const sofa::core::ExecParams* params);

    virtual Result processNodeTopDown(simulation::Node* node);

    void processNodeBottomUp(simulation::Node* node);

};


}
}


#endif
