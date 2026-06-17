#pragma once
#include <sofa/component/integrationschemes/backward/LinearMultistepIntegrationScheme.h>
SOFA_HEADER_DEPRECATED("v26.06", "v27.06", "sofa/component/integrationschemes/backward/LinearMultistepIntegrationScheme.h")

namespace sofa::component::odesolver::backward
{
using BaseLinearMultiStepMethod SOFA_ATTRIBUTE_DEPRECATED("v26.06", "v27.06",
    "sofa::component::odesolver::backward::BaseLinearMultiStepMethod has been renamed to "
    "sofa::component::integrationschemes::backward::LinearMultistepIntegrationScheme")
    = sofa::component::integrationschemes::backward::LinearMultistepIntegrationScheme;
} // namespace sofa::component::odesolver::backward
