#pragma once
#include <sofa/component/integrationschemes/forward/RungeKutta4IntegrationScheme.h>
SOFA_HEADER_DEPRECATED("v26.06", "v27.06", "sofa/component/integrationschemes/forward/RungeKutta4IntegrationScheme.h")

namespace sofa::component::odesolver::forward
{
using RungeKutta4Solver SOFA_ATTRIBUTE_DEPRECATED("v26.06", "v27.06",
    "sofa::component::odesolver::forward::RungeKutta4Solver has been renamed to "
    "sofa::component::integrationschemes::forward::RungeKutta4IntegrationScheme")
    = sofa::component::integrationschemes::forward::RungeKutta4IntegrationScheme;
} // namespace sofa::component::odesolver::forward
