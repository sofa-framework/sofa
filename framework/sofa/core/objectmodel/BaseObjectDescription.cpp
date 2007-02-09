#include "BaseObjectDescription.h"
#include "BaseContext.h"
#include "BaseObject.h"
#include <iostream>

namespace sofa
{

namespace core
{

namespace objectmodel
{

BaseObjectDescription::~BaseObjectDescription()
{
}

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
        std::cout << "Found node "<<nodeName<<": "<<node->getName()<<std::endl;
        Base* obj = node->getObject();
        BaseContext* ctx = dynamic_cast<BaseContext*>(obj);
        if (ctx != NULL)
        {
            std::cout << "Node "<<nodeName<<" is a context, returning MechanicalState."<<std::endl;
            obj = ctx->getMechanicalState();
        }
        return obj;
    }
    else
    {
        std::cout << "Node "<<nodeName<<" NOT FOUND."<<std::endl;
        return NULL;
    }
}

} // namespace objectmodel

} // namespace core

} // namespace sofa
