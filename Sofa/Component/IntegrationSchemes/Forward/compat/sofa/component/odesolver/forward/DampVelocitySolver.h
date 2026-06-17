#pragma once
#include <sofa/component/integrationschemes/forward/DampVelocityIntegrationScheme.h>
SOFA_HEADER_DEPRECATED("v26.06", "v27.06", "sofa/component/integrationschemes/forward/DampVelocityIntegrationScheme.h")

namespace sofa::component::odesolver::forward
{
using DampVelocitySolver SOFA_ATTRIBUTE_DEPRECATED("v26.06", "v27.06",
    "sofa::component::odesolver::forward::DampVelocitySolver has been renamed to "
    "sofa::component::integrationschemes::forward::DampVelocityIntegrationScheme")
    = sofa::component::integrationschemes::forward::DampVelocityIntegrationScheme;
} // namespace sofa::component::odesolver::forward
