#include <sofa/core/fwd.h>
#include <sofa/core/topology/TopologyChange.h>
#include <sofa/core/BaseState.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include <sofa/core/objectmodel/ContextObject.h>

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

#include <sofa/core/topology/Topology.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/topology/BaseTopology.h>

#include <sofa/core/collision/Pipeline.h>

#include <sofa/core/visual/VisualLoop.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/visual/VisualManager.h>
#include <sofa/core/visual/Shader.h>

#include <sofa/core/collision/Pipeline.h>
#include <sofa/core/collision/Pipeline.h>

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

#define DEFINE_OPAQUE_FUNCTION_FOR(TYPENAME) \
    template<> \
    TYPENAME* castBaseTo(sofa::core::objectmodel::Base* base) \
    { return dynamic_cast<TYPENAME*>(base); } \
    sofa::core::objectmodel::Base* baseFrom(TYPENAME* b) \
    { return dynamic_cast<sofa::core::objectmodel::Base*>(b); } \
    namespace objectmodel::base { template<> const sofa::core::objectmodel::BaseClass* GetClass<TYPENAME>() \
    { return TYPENAME::GetClass(); } }

DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::BaseState);
DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::BaseMapping);
DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::BehaviorModel);
DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::CollisionModel);

DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::objectmodel::BaseObject);
DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::objectmodel::ContextObject);
DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::objectmodel::ConfigurationSetting);

DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::behavior::BaseAnimationLoop);
DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::behavior::BaseMass);
DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::behavior::OdeSolver);
DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::behavior::ConstraintSolver);
DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::behavior::BaseLinearSolver);
DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::behavior::BaseMechanicalState);
DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::behavior::BaseForceField);
DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::behavior::BaseInteractionForceField);
DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::behavior::BaseProjectiveConstraintSet);
DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::behavior::BaseConstraintSet);

DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::topology::Topology);
DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::topology::BaseMeshTopology);
DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::topology::BaseTopologyObject);

DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::collision::Pipeline);
DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::visual::VisualLoop);
DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::visual::Shader);
DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::visual::VisualModel);
DEFINE_OPAQUE_FUNCTION_FOR(sofa::core::visual::VisualManager);

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

