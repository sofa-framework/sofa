#pragma once
#include <sofa/component/integrationschemes/backward/StaticEquilibriumIntegrationScheme.h>
SOFA_HEADER_DEPRECATED("v26.06", "v27.06", "sofa/component/integrationschemes/backward/StaticEquilibriumIntegrationScheme.h")

namespace sofa::component::odesolver::backward
{
using StaticSolver SOFA_ATTRIBUTE_DEPRECATED("v26.06", "v27.06",
    "sofa::component::odesolver::backward::StaticSolver has been renamed to "
    "sofa::component::integrationschemes::backward::StaticEquilibriumIntegrationScheme")
    = sofa::component::integrationschemes::backward::StaticEquilibriumIntegrationScheme;
} // namespace sofa::component::odesolver::backward
