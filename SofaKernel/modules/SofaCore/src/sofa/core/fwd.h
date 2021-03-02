#pragma once
#include <sofa/core/config.h>
#include <iosfwd>

namespace sofa::core
{
class ExecParams;
class ConstraintParams;
class BaseMapping;
class CollisionModel;
class CollisionElementIterator;
class ConstraintParams;
class ExecParams;
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

namespace sofa::core::visual
{
class VisualParams;

class FlagTreeItem;
SOFA_CORE_API std::ostream& operator<< ( std::ostream& os, const FlagTreeItem& root );
SOFA_CORE_API std::istream& operator>> ( std::istream& in, FlagTreeItem& root );

class DisplayFlags;
SOFA_CORE_API std::ostream& operator<< ( std::ostream& os, const DisplayFlags& flags );
SOFA_CORE_API std::istream& operator>> ( std::istream& in, DisplayFlags& flags );
}

namespace sofa::component::topology
{
class TetrahedronSetTopologyContainer;
SOFA_CORE_API std::ostream& operator<< (std::ostream& out, const TetrahedronSetTopologyContainer& t);
SOFA_CORE_API std::istream& operator>>(std::istream& in, TetrahedronSetTopologyContainer& t);
}

namespace sofa::core::objectmodel::basecontext
{
SOFA_CORE_API SReal getDt(sofa::core::objectmodel::BaseContext* context);
SOFA_CORE_API SReal getTime(sofa::core::objectmodel::BaseContext* context);
}
