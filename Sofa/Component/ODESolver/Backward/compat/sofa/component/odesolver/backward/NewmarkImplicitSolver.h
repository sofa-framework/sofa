#pragma once
#include <sofa/component/integrationscheme/backward/NewmarkIntegrationScheme.h>
SOFA_HEADER_DEPRECATED("v26.06", "v27.06", "sofa/component/integrationscheme/backward/NewmarkIntegrationScheme.h")

namespace sofa::component::odesolver::backward
{
using NewmarkImplicitSolver SOFA_ATTRIBUTE_DEPRECATED("v26.06", "v27.06",
    "sofa::component::odesolver::backward::NewmarkImplicitSolver has been renamed to "
    "sofa::component::integrationscheme::backward::NewmarkIntegrationScheme")
    = sofa::component::integrationscheme::backward::NewmarkIntegrationScheme;
} // namespace sofa::component::odesolver::backward
