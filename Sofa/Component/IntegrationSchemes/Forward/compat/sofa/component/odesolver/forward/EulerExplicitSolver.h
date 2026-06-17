#pragma once
#include <sofa/component/integrationschemes/forward/EulerExplicitIntegrationScheme.h>
SOFA_HEADER_DEPRECATED("v26.06", "v27.06", "sofa/component/integrationschemes/forward/EulerExplicitIntegrationScheme.h")

namespace sofa::component::odesolver::forward
{
using EulerExplicitSolver SOFA_ATTRIBUTE_DEPRECATED("v26.06", "v27.06",
    "sofa::component::odesolver::forward::EulerExplicitSolver has been renamed to "
    "sofa::component::integrationschemes::forward::EulerExplicitIntegrationScheme")
    = sofa::component::integrationschemes::forward::EulerExplicitIntegrationScheme;
} // namespace sofa::component::odesolver::forward
