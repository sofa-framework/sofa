#pragma once
#include <sofa/component/integrationscheme/backward/StaticEquilibriumIntegrationScheme.h>
SOFA_HEADER_DEPRECATED("v26.06", "v27.06", "sofa/component/integrationscheme/backward/StaticEquilibriumIntegrationScheme.h")

namespace sofa::component::odesolver::backward
{
using StaticSolver SOFA_ATTRIBUTE_DEPRECATED("v26.06", "v27.06",
    "sofa::component::odesolver::backward::StaticSolver has been renamed to "
    "sofa::component::integrationscheme::backward::StaticEquilibriumIntegrationScheme")
    = sofa::component::integrationscheme::backward::StaticEquilibriumIntegrationScheme;
} // namespace sofa::component::odesolver::backward
