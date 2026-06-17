#pragma once
#include <sofa/component/integrationschemes/backward/NewmarkIntegrationScheme.h>
SOFA_HEADER_DEPRECATED("v26.06", "v27.06", "sofa/component/integrationschemes/backward/NewmarkIntegrationScheme.h")

namespace sofa::component::odesolver::backward
{
using NewmarkImplicitSolver SOFA_ATTRIBUTE_DEPRECATED("v26.06", "v27.06",
    "sofa::component::odesolver::backward::NewmarkImplicitSolver has been renamed to "
    "sofa::component::integrationschemes::backward::NewmarkIntegrationScheme")
    = sofa::component::integrationschemes::backward::NewmarkIntegrationScheme;
} // namespace sofa::component::odesolver::backward
