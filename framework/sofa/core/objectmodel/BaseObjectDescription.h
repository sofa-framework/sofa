#ifndef SOFA_CORE_OBJECTMODEL_BASEOBJECTDESCRIPTION_H
#define SOFA_CORE_OBJECTMODEL_BASEOBJECTDESCRIPTION_H

#include <string>
#include <list>
#include <map>

namespace sofa
{

namespace core
{

namespace objectmodel
{

class Base;

class BaseObjectDescription
{
public:
    virtual ~BaseObjectDescription();

    /// Get the associated object
    virtual Base* getObject() = 0;

    /// Get the node instance name
    virtual const std::string& getName() const = 0;

    /// Get the parent node
    virtual BaseObjectDescription* getParent() const = 0;

    typedef std::map<std::string,std::string*> AttributeMap;

    /// Get all attribute data, read-only
    virtual const AttributeMap& getAttributeMap() const = 0;

    /// Find a node given its name
    virtual BaseObjectDescription* find(const char* nodeName, bool absolute=false) = 0;

    /// Get an attribute given its name (return defaultVal if not present)
    virtual const char* getAttribute(const std::string& attr, const char* defaultVal=NULL);

    /// Get an attribute given its name (return defaultVal if not present)
    virtual void removeAttribute(const std::string&)
    {
    }

    virtual std::string getFullName() const;

    /// Find an object given its name
    virtual Base* findObject(const char* nodeName);

};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
