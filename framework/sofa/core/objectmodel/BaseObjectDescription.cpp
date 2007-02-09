#include "BaseObjectDescription.h"


namespace sofa
{

namespace core
{

namespace objectmodel
{

/// Get an attribute given its name (return defaultVal if not present)
const char* BaseObjectDescription::getAttribute(const std::string& attr, const char* defaultVal)
{
    const AttributeMap& map = this->getAttributeMap();
    AttributeMap::const_iterator it = map.find(attr);
    if (it == map.end())
        return defaultVal;
    else
        return it->second->c_str();
}

std::string BaseObjectDescription::getFullName() const
{
    BaseObjectDescription* parent = getParent();
    if (parent==NULL) return "/";
    std::string pname = parent->getFullName();
    pname += "/";
    pname += getName();
    return pname;
}

/// Find an object given its name
Base* BaseObjectDescription::findObject(const char* nodeName)
{
    BaseObjectDescription* node = find(nodeName);
    if (node!=NULL)
    {
        //std::cout << "Found node "<<nodeName<<": "<<node->getName()<<std::endl;
        return node->getObject();
    }
    else return NULL;
}

} // namespace objectmodel

} // namespace core

} // namespace sofa
