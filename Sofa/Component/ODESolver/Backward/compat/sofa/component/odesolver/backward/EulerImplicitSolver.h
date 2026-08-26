#pragma once
#include <sofa/component/integrationscheme/backward/EulerImplicitIntegrationScheme.h>
SOFA_HEADER_DEPRECATED("v26.06", "v27.06", "sofa/component/integrationscheme/backward/EulerImplicitIntegrationScheme.h")

namespace sofa::component::odesolver::backward
{
using EulerImplicitSolver SOFA_ATTRIBUTE_DEPRECATED("v26.06", "v27.06",
    "sofa::component::odesolver::backward::EulerImplicitSolver has been renamed to "
    "sofa::component::integrationscheme::backward::EulerImplicitIntegrationScheme")
    = sofa::component::integrationscheme::backward::EulerImplicitIntegrationScheme;
} // namespace sofa::component::odesolver::backward
