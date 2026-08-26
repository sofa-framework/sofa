#pragma once
#include <sofa/component/integrationscheme/backward/LinearMultistepIntegrationScheme.h>
SOFA_HEADER_DEPRECATED("v26.06", "v27.06", "sofa/component/integrationscheme/backward/LinearMultistepIntegrationScheme.h")

namespace sofa::component::odesolver::backward
{
using BaseLinearMultiStepMethod SOFA_ATTRIBUTE_DEPRECATED("v26.06", "v27.06",
    "sofa::component::odesolver::backward::BaseLinearMultiStepMethod has been renamed to "
    "sofa::component::integrationscheme::backward::LinearMultistepIntegrationScheme")
    = sofa::component::integrationscheme::backward::LinearMultistepIntegrationScheme;
} // namespace sofa::component::odesolver::backward
