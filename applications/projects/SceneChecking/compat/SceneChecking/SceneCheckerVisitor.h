#pragma once
#include <sofa/simulation/SceneCheckerVisitor.h>
SOFA_HEADER_DEPRECATED("v26.06", "v27.06", "sofa/simulation/SceneCheckerVisitor.h")

namespace sofa::scenechecking
{
using SceneCheckerVisitor SOFA_ATTRIBUTE_DEPRECATED("v26.06", "v27.06",
    "sofa::scenechecking::SceneCheckerVisitor has been moved to sofa::simulation::SceneCheckerVisitor")
    = sofa::simulation::SceneCheckerVisitor;
} // namespace sofa::scenechecking
