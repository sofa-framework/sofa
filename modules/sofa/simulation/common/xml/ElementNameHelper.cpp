#include <sofa/simulation/common/xml/ElementNameHelper.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace simulation
{
namespace xml
{


ElementNameHelper::ElementNameHelper()
{

}

ElementNameHelper::~ElementNameHelper()
{

}

std::string ElementNameHelper::resolveName(const std::string& type, const std::string& name)
{
    std::string resolvedName;
    if(name.empty())
    {
        std::string radix = core::ObjectFactory::ShortName(type);
        registerName(radix);
        std::ostringstream oss;
        oss << radix << instanceCounter[radix];
        resolvedName = oss.str();
    }
    else
    {
        resolvedName = name;
    }
    return resolvedName;

}

void ElementNameHelper::registerName(const std::string& name)
{
    if( instanceCounter.find(name) != instanceCounter.end())
    {
        instanceCounter[name]++;
    }
    else
    {
        instanceCounter[name] = 1;
    }
}

}
}
}
