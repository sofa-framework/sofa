#pragma once
#include <sofa/core/config.h>
#include <iosfwd>
namespace sofa::core::objectmodel
{
    class Base;
    class BaseObject;
    class BaseNode;
    class BaseContext;
    class BaseData;
    class BaseLink;
}

namespace sofa::core::topology
{
    class TopologyChange;

    /// Output  stream
    std::ostream& operator<< ( std::ostream& out, const sofa::core::topology::TopologyChange* t );

    /// Input (empty) stream
    std::istream& operator>> ( std::istream& in, sofa::core::topology::TopologyChange*& t );

    /// Input (empty) stream
    std::istream& operator>> ( std::istream& in, const sofa::core::topology::TopologyChange*& );

}
