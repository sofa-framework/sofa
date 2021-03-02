#include <sofa/core/fwd.h>
#include <sofa/core/topology/TopologyChange.h>

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

