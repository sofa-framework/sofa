#pragma once
#include <sofa/core/config.h>
#include <iosfwd>

namespace sofa::core
{
class BaseState;
class ExecParams;
class ConstraintParams;
class BaseMapping;
class CollisionModel;
class CollisionElementIterator;
class ConstraintParams;
class BehaviorModel;
}

namespace sofa::core::objectmodel
{
class Base;
class BaseObject;
class BaseNode;
class BaseContext;
class BaseData;
class BaseLink;
class BaseClass;
class AbstractDataLink;
class Event;
class BaseClass;
class AbstractDataLink;
class ContextObject;
class ConfigurationSetting;


class Tag;
SOFA_CORE_API std::ostream& operator<<(std::ostream& o, const Tag& t);
SOFA_CORE_API std::istream& operator>>(std::istream& i, Tag& t);
}


namespace sofa::core::behavior
{
class BaseForceField;
class BaseMass;
class BaseMechanicalState;
class BaseAnimationLoop;
class BaseConstraint;
class BaseConstraintSet;
class ConstraintSolver;
class ConstraintResolution;
class OdeSolver;
class BaseLinearSolver;
class BaseInteractionForceField;
class BaseProjectiveConstraintSet;

template<class T>
class MechanicalState;
}

namespace sofa::core::topology
{
class BaseMeshTopology;
class BaseTopologyObject;
class Topology;
class TopologyEngine;
class TopologyChange;
SOFA_CORE_API std::ostream& operator<< ( std::ostream& out, const sofa::core::topology::TopologyChange* t );
SOFA_CORE_API std::istream& operator>> ( std::istream& in, sofa::core::topology::TopologyChange*& t );
SOFA_CORE_API std::istream& operator>> ( std::istream& in, const sofa::core::topology::TopologyChange*& );
}

namespace sofa::core::visual
{
class VisualParams;
class VisualLoop;
class VisualModel;
class VisualManager;
class Shader;

class FlagTreeItem;
SOFA_CORE_API std::ostream& operator<< ( std::ostream& os, const FlagTreeItem& root );
SOFA_CORE_API std::istream& operator>> ( std::istream& in, FlagTreeItem& root );

class DisplayFlags;
SOFA_CORE_API std::ostream& operator<< ( std::ostream& os, const DisplayFlags& flags );
SOFA_CORE_API std::istream& operator>> ( std::istream& in, DisplayFlags& flags );
}

namespace sofa::core::collision
{
class Pipeline;
}

namespace sofa::core
{
/////////////////////////////// CORE::OPAQUE FUNCTION /////////////////////////////////////////////////
///
/// CORE::OPAQUE function are a groupe of function that make "opaque" some of the common sofa behaviors.
///
/// Core::Opaque functions are:
///     - Base* sofa::core::baseFrom(T*) replace dynamic_cast<Base*>(T*);
///     - T* sofa::core::castBaseTo(Base*) replace dynamic_cast<T*>(Base*);
///     - sofa:core::base::GetClass<T>() replace T::GetClass();
///
/// These functions are called "opaque" as they work with only forward declaration of the involved
/// types in comparison to class methods the requires the full class declaration to be used.
///
/// It is highly recommanded to use as much as possible opaque function in header files as this
/// allow to reduce the dependency tree.
///
/// Opaque function may be slower at runtime (by one function call) but this is true only if LTO isn't
/// able to optimize them properly. If you have experience/feedback with LTO please join the discussion
/// in https://github.com/sofa-framework/sofa/discussions/1822
///////////////////////////////////////////////////////////////////////////////////////////////////////
template<class B>
sofa::core::objectmodel::Base* baseFrom(B*b){ return dynamic_cast<sofa::core::objectmodel::Base*>(b);}

template<class Dest>
Dest castBaseTo(sofa::core::objectmodel::Base* base){ return dynamic_cast<Dest>(base); }

/// getClass is in the namespace base as it is an opaque function for Base::GetClass
namespace base
{
template<class B>
const sofa::core::objectmodel::BaseClass* GetClass(){return B::GetClass(); }
}

/// Macro use to simplify the declaration of the core opaque function.
#define DECLARE_OPAQUE_FUNCTION_FOR(TYPENAME) \
    template<> SOFA_CORE_API TYPENAME* castBaseTo(sofa::core::objectmodel::Base* base); \
    SOFA_CORE_API sofa::core::objectmodel::Base* baseFrom(TYPENAME* b); \
    namespace base { template<> SOFA_CORE_API const sofa::core::objectmodel::BaseClass* GetClass<TYPENAME>(); }

DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::BaseState);
DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::BaseMapping);
DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::BehaviorModel);
DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::CollisionModel);

DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::objectmodel::BaseObject);
DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::objectmodel::ContextObject);
DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::objectmodel::ConfigurationSetting);

DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::behavior::BaseAnimationLoop);
DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::behavior::BaseMass);
DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::behavior::OdeSolver);
DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::behavior::ConstraintSolver);
DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::behavior::BaseLinearSolver);
DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::behavior::BaseMechanicalState);
DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::behavior::BaseForceField);
DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::behavior::BaseInteractionForceField);
DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::behavior::BaseProjectiveConstraintSet);
DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::behavior::BaseConstraintSet);

DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::topology::Topology);
DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::topology::BaseMeshTopology);
DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::topology::BaseTopologyObject);

DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::collision::Pipeline);
DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::visual::VisualLoop);
DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::visual::Shader);
DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::visual::VisualModel);
DECLARE_OPAQUE_FUNCTION_FOR(sofa::core::visual::VisualManager);
}

namespace sofa::core::objectmodel::basecontext
{
SOFA_CORE_API SReal getDt(sofa::core::objectmodel::BaseContext* context);
SOFA_CORE_API SReal getTime(sofa::core::objectmodel::BaseContext* context);
}

namespace sofa::core::objectmodel::basecontext
{
SOFA_CORE_API SReal getDt(sofa::core::objectmodel::BaseContext* context);
SOFA_CORE_API SReal getTime(sofa::core::objectmodel::BaseContext* context);
}
