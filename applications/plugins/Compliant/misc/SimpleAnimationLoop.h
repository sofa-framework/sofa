#ifndef COMPLIANT_SIMPLEANIMATIONLOOP_H
#define COMPLIANT_SIMPLEANIMATIONLOOP_H

#include <sofa/core/behavior/BaseAnimationLoop.h>
#include <Compliant/config.h>

namespace sofa
{

namespace simulation
{

class SOFA_Compliant_API SimpleAnimationLoop : public sofa::core::behavior::BaseAnimationLoop {

  public:

    SOFA_CLASS(SimpleAnimationLoop, sofa::core::behavior::BaseAnimationLoop);

  public:

    SimpleAnimationLoop();
    
    Data<unsigned> extra_steps;

    /// perform one animation step
    void step(const core::ExecParams* params, SReal dt) override;
    
};

}
}



#endif
