#pragma once
#include <sofa/component/integrationschemes/backward/BDFIntegrationScheme.h>
SOFA_HEADER_DEPRECATED("v26.06", "v27.06", "sofa/component/integrationschemes/backward/BDFIntegrationScheme.h")

namespace sofa::component::odesolver::backward
{
using BDFOdeSolver SOFA_ATTRIBUTE_DEPRECATED("v26.06", "v27.06",
    "sofa::component::odesolver::backward::BDFOdeSolver has been renamed to "
    "sofa::component::integrationschemes::backward::BDFIntegrationScheme")
    = sofa::component::integrationschemes::backward::BDFIntegrationScheme;
} // namespace sofa::component::odesolver::backward
