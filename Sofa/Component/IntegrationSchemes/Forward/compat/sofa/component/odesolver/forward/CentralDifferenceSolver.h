#pragma once
#include <sofa/component/integrationschemes/forward/CentralDifferenceIntegrationScheme.h>
SOFA_HEADER_DEPRECATED("v26.06", "v27.06", "sofa/component/integrationschemes/forward/CentralDifferenceIntegrationScheme.h")

namespace sofa::component::odesolver::forward
{
using CentralDifferenceSolver SOFA_ATTRIBUTE_DEPRECATED("v26.06", "v27.06",
    "sofa::component::odesolver::forward::CentralDifferenceSolver has been renamed to "
    "sofa::component::integrationschemes::forward::CentralDifferenceIntegrationScheme")
    = sofa::component::integrationschemes::forward::CentralDifferenceIntegrationScheme;
} // namespace sofa::component::odesolver::forward
