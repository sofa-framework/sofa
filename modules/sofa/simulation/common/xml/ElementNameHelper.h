#ifndef SOFA_SIMULATION_XML_ELEMENTNAMEHELPER
#define SOFA_SIMULATION_XML_ELEMENTNAMEHELPER


#include <map>
#include <string>

namespace sofa
{
namespace simulation
{
namespace xml
{


class ElementNameHelper
{
protected:
    std::map<std::string, int> instanceCounter;
    void registerName(const std::string& name);

public:
    ElementNameHelper();
    ~ElementNameHelper(); //terminal class.

    std::string resolveName(const std::string& type, const std::string& name);
};

}
}
}


#endif // SOFA_SIMULATION_XML_ELEMENTNAMEHELPER
