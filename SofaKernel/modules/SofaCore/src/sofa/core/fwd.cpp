#include <sofa/core/fwd.h>
#include <sofa/core/topology/TopologyChange.h>
#include <sofa/core/BaseState.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/core/CollisionModel.h>

#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/BehaviorModel.h>
#include <sofa/core/behavior/BaseAnimationLoop.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/behavior/ConstraintSolver.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/behavior/BaseConstraintSet.h>
#include <sofa/core/behavior/BaseInteractionForceField.h>
#include <sofa/core/behavior/BaseInteractionConstraint.h>
#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>
#include <sofa/core/behavior/BaseConstraint.h>

#include <sofa/core/collision/Pipeline.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include <sofa/core/objectmodel/ContextObject.h>

#include <sofa/core/topology/Topology.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/topology/BaseTopology.h>

#include <sofa/core/visual/VisualLoop.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/visual/VisualManager.h>
#include <sofa/core/visual/Shader.h>

namespace sofa::core::topology
{

std::ostream& operator<< ( std::ostream& out, const TopologyChange* t )
{
    if (t)
    {
        t->write(out);
    }
    return out;
}

/// Input (empty) stream
std::istream& operator>> ( std::istream& in, TopologyChange*& t )
{
    if (t)
    {
        t->read(in);
    }
    return in;
}

/// Input (empty) stream
std::istream& operator>> ( std::istream& in, const TopologyChange*& )
{
    return in;
}

}

namespace sofa::core
{

SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::BaseState);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::BaseMapping);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::BehaviorModel);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::CollisionModel);

SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::objectmodel::BaseObject);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::objectmodel::ContextObject);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::objectmodel::ConfigurationSetting);

SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::behavior::BaseAnimationLoop);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::behavior::BaseMass);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::behavior::OdeSolver);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::behavior::ConstraintSolver);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::behavior::BaseLinearSolver);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::behavior::BaseMechanicalState);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::behavior::BaseForceField);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::behavior::BaseInteractionForceField);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::behavior::BaseProjectiveConstraintSet);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::behavior::BaseConstraintSet);

SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::topology::Topology);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::topology::BaseMeshTopology);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::topology::BaseTopologyObject);

SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::collision::Pipeline);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::visual::VisualLoop);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::visual::Shader);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::visual::VisualModel);
SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(sofa::core::visual::VisualManager);

}

namespace sofa::core::objectmodel::basecontext
{

SReal getDt(sofa::core::objectmodel::BaseContext* context)
{
    return context->getDt();
}

SReal getTime(sofa::core::objectmodel::BaseContext* context)
{
    return context->getTime();
}

}

