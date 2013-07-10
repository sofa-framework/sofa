#include "PostStabilization.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa {
namespace component {
namespace odesolver {


SOFA_DECL_CLASS(PostStabilization);
int PostStabilizationClass = core::RegisterObject("Post-stabilization of kinematic constraints").add< PostStabilization >();

}
}
}
