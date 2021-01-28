#pragma once
#include <sofa/core/config.h>
#include <iosfwd>

namespace sofa::core
{
    class BaseMapping;
    class ConstraintParams;
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
}

namespace sofa::core::topology
{
    class Topology;
    class TopologyChange;

    /// Output  stream
    SOFA_CORE_API std::ostream& operator<< ( std::ostream& out, const sofa::core::topology::TopologyChange* t );

    /// Input (empty) stream
    SOFA_CORE_API std::istream& operator>> ( std::istream& in, sofa::core::topology::TopologyChange*& t );

    /// Input (empty) stream
    SOFA_CORE_API std::istream& operator>> ( std::istream& in, const sofa::core::topology::TopologyChange*& );
}

namespace sofa::core::visual
{
    class VisualParams;
    class DisplayFlags;
}
