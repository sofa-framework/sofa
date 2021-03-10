#pragma once
#include <sofa/core/config.h>
#include <iosfwd>

namespace sofa::helper::visual { class DrawTool; }

namespace sofa::core
{

    class BaseMapping;
    class ConstraintParams;
    class ExecParams;
    class MechanicalParams;
    class CollisionElementIterator;
    class CollisionModel;
    class ConstraintParams;

    namespace execparams
    {
        SOFA_CORE_API ExecParams* defaultInstance();
    }

    namespace constraintparams
    {
        SOFA_CORE_API const ConstraintParams* defaultInstance();
        SOFA_CORE_API ExecParams* dynamicCastToExecParams(sofa::core::ConstraintParams*);
        SOFA_CORE_API const ExecParams* dynamicCastToExecParams(const sofa::core::ConstraintParams*);
    }

    namespace mechanicalparams
    {
        SOFA_CORE_API const MechanicalParams* defaultInstance();
        SOFA_CORE_API ExecParams* dynamicCastToExecParams(sofa::core::MechanicalParams*);
        SOFA_CORE_API const ExecParams* dynamicCastToExecParams(const sofa::core::MechanicalParams*);

        SOFA_CORE_API SReal kFactor(const sofa::core::MechanicalParams*);
        SOFA_CORE_API SReal bFactor(const sofa::core::MechanicalParams*);
        SOFA_CORE_API SReal kFactorIncludingRayleighDamping(const sofa::core::MechanicalParams*, SReal d);
        SOFA_CORE_API SReal mFactorIncludingRayleighDamping(const sofa::core::MechanicalParams*, SReal d);
        SOFA_CORE_API SReal dt(const sofa::core::MechanicalParams*);
    }
}

namespace sofa::core::objectmodel
{
class Base;
class BaseObject;
class BaseNode;
class BaseContext;
class BaseData;
class BaseLink;
class BaseNode;
class Event;

class Tag;
SOFA_CORE_API std::ostream& operator<<(std::ostream& o, const Tag& t);
SOFA_CORE_API std::istream& operator>>(std::istream& i, Tag& t);
}

namespace sofa::core::behavior
{
class BaseForceField;
class BaseMass;
class BaseMechanicalState;
class BaseConstraint;
class BaseConstraintSet;
class ConstraintSolver;
class ConstraintResolution;

template<class T>
class MechanicalState;
}

namespace sofa::core::topology
{
class TopologyChange;
SOFA_CORE_API std::ostream& operator<< ( std::ostream& out, const sofa::core::topology::TopologyChange* t );
SOFA_CORE_API std::istream& operator>> ( std::istream& in, sofa::core::topology::TopologyChange*& t );
SOFA_CORE_API std::istream& operator>> ( std::istream& in, const sofa::core::topology::TopologyChange*& );

class Topology;
}


namespace sofa::component::topology
{
class TetrahedronSetTopologyContainer;
SOFA_CORE_API std::ostream& operator<< (std::ostream& out, const TetrahedronSetTopologyContainer& t);
SOFA_CORE_API std::istream& operator>>(std::istream& in, TetrahedronSetTopologyContainer& t);
}


namespace sofa::core::visual
{
class FlagTreeItem;
SOFA_CORE_API std::ostream& operator<< ( std::ostream& os, const FlagTreeItem& root );
SOFA_CORE_API std::istream& operator>> ( std::istream& in, FlagTreeItem& root );

class DisplayFlags;
SOFA_CORE_API std::ostream& operator<< ( std::ostream& os, const DisplayFlags& flags );
SOFA_CORE_API std::istream& operator>> ( std::istream& in, DisplayFlags& flags );

using sofa::helper::visual::DrawTool;

class VisualParams;
namespace visualparams
{
    SOFA_CORE_API VisualParams* defaultInstance();

    SOFA_CORE_API ExecParams* dynamicCastToExecParams(sofa::core::visual::VisualParams*);
    SOFA_CORE_API const ExecParams* dynamicCastToExecParams(const sofa::core::visual::VisualParams*);

    SOFA_CORE_API sofa::core::visual::DrawTool* getDrawTool(VisualParams* params);
    SOFA_CORE_API sofa::core::visual::DisplayFlags& getDisplayFlags(VisualParams* params);
    SOFA_CORE_API sofa::core::visual::DrawTool* getDrawTool(const VisualParams* params);
    SOFA_CORE_API const sofa::core::visual::DisplayFlags& getDisplayFlags(const VisualParams* params);
}

}

namespace sofa::core::objectmodel::basecontext
{
SOFA_CORE_API SReal getDt(sofa::core::objectmodel::BaseContext* context);
SOFA_CORE_API SReal getTime(sofa::core::objectmodel::BaseContext* context);
}
