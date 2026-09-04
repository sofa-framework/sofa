#pragma once
#include <sofa/component/integrationscheme/forward/DampVelocityIntegrationScheme.h>
SOFA_HEADER_DEPRECATED("v26.06", "v27.06", "sofa/component/integrationscheme/forward/DampVelocityIntegrationScheme.h")

namespace sofa::component::odesolver::forward
{
using DampVelocitySolver SOFA_ATTRIBUTE_DEPRECATED("v26.06", "v27.06",
    "sofa::component::odesolver::forward::DampVelocitySolver has been renamed to "
    "sofa::component::integrationscheme::forward::DampVelocityIntegrationScheme")
    = sofa::component::integrationscheme::forward::DampVelocityIntegrationScheme;
} // namespace sofa::component::odesolver::forward
