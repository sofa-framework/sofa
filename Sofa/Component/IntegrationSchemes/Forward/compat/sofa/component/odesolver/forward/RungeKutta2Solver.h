#pragma once
#include <sofa/component/integrationschemes/forward/RungeKutta2IntegrationScheme.h>
SOFA_HEADER_DEPRECATED("v26.06", "v27.06", "sofa/component/integrationschemes/forward/RungeKutta2IntegrationScheme.h")

namespace sofa::component::odesolver::forward
{
using RungeKutta2Solver SOFA_ATTRIBUTE_DEPRECATED("v26.06", "v27.06",
    "sofa::component::odesolver::forward::RungeKutta2Solver has been renamed to "
    "sofa::component::integrationschemes::forward::RungeKutta2IntegrationScheme")
    = sofa::component::integrationschemes::forward::RungeKutta2IntegrationScheme;
} // namespace sofa::component::odesolver::forward
