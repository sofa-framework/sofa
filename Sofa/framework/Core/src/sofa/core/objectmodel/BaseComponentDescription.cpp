/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/objectmodel/BaseComponentDescription.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/BaseComponent.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <iostream>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/system/Locale.h>

namespace sofa::core::objectmodel
{

BaseComponentDescription::BaseComponentDescription(const char* name, const char* type)
{
    if (name)
        attributes["name"] = name;
    if (type)
        attributes["type"] = type;
}

BaseComponentDescription::~BaseComponentDescription()
{
    attributes.clear();
}

/// Get the associated object (or nullptr if it is not created yet)
Base* BaseComponentDescription::getObject()
{
    return nullptr;
}

/// Get the object instance name
std::string BaseComponentDescription::getName()
{
    return std::string(getAttribute("name",""));
}

void BaseComponentDescription::setName(const std::string& name)
{
    setAttribute("name",name);
}

/// Get the parent node
BaseComponentDescription* BaseComponentDescription::getParent() const
{
    return nullptr;
}

/// Get the file where this description was read from. Useful to resolve relative file paths.
std::string BaseComponentDescription::getBaseFile()
{
    return "";
}

///// Get all attribute data, read-only
const BaseComponentDescription::AttributeMap& BaseComponentDescription::getAttributeMap() const
{
    return attributes;
}

/// Find an object description given its name (relative to this object)
BaseComponentDescription* BaseComponentDescription::find(const char* /*nodeName*/, bool /*absolute*/)
{
    return nullptr;
}

/// Remove an attribute given its name, returns false if the attribute was not there.
bool BaseComponentDescription::removeAttribute(const std::string& attr)
{
    const AttributeMap::iterator it = attributes.find(attr);
    if (it == attributes.end())
        return false;

    attributes.erase(it);
    return true;
}

/// Get an attribute given its name (return defaultVal if not present)
const char* BaseComponentDescription::getAttribute(const std::string& attr, const char* defaultVal)
{
    const AttributeMap::iterator it = attributes.find(attr);
    if (it == attributes.end())
        return defaultVal;
    else
        return it->second.c_str();
}

/// Docs is in .h
float BaseComponentDescription::getAttributeAsFloat(const std::string& attr, const float defaultVal)
{
    const AttributeMap::iterator it = attributes.find(attr);
    if (it == attributes.end())
        return defaultVal;

    // Make sure that strtof uses a dot '.' as the decimal separator.
    helper::system::TemporaryLocale locale(LC_NUMERIC, "C");

    const char* attrstr=it->second.c_str();
    char* end=nullptr;
    const float retval = strtof(attrstr, &end);

    /// It is important to check that the attribute was totally parsed to report
    /// message to users because a silent error is the worse thing that can happen in UX.
    if(end !=  attrstr+strlen(attrstr)){
        std::stringstream msg;
        msg << "Unable to parse a float value from attribute '" << attr << "'='"<<it->second.c_str()<<"'. "
               "Use the default value '"<<defaultVal<< "' instead.";
        errors.push_back(msg.str());
        return defaultVal ;
    }

    return retval ;
}

/// Docs is in .h
int BaseComponentDescription::getAttributeAsInt(const std::string& attr, const int defaultVal)
{
    const AttributeMap::iterator it = attributes.find(attr);
    if (it == attributes.end())
        return defaultVal;

    const char* attrstr=it->second.c_str();
    char* end=nullptr;
    const int retval = static_cast<int>(strtol(attrstr, &end, 10));

    /// It is important to check that the attribute was totally parsed to report
    /// message to users because a silent error is the worse thing that can happen in UX.
    if(end !=  attrstr+strlen(attrstr)){
        std::stringstream msg;
        msg << "Unable to parse an integer value from attribute '" << attr << "'='"<<it->second.c_str()<<"'. "
               "Use the default value '"<<defaultVal<< "' instead.";
        errors.push_back(msg.str());
        return defaultVal;
    }

    return retval ;
}

/// Set an attribute. Override any existing value
void BaseComponentDescription::setAttribute(const std::string& attr, const std::string &val)
{
    attributes[attr] = val;
}

std::string BaseComponentDescription::getFullName()
{
    BaseComponentDescription* parent = getParent();
    if (parent==nullptr) return "/";
    std::string pname = parent->getFullName();
    pname += "/";
    pname += getName();
    return pname;
}

/// Find an object given its name
Base* BaseComponentDescription::findObject(const char* nodeName)
{
    BaseComponentDescription* node = find(nodeName);
    if (node!=nullptr)
    {
        Base* obj = node->getObject();
        const BaseContext* ctx = obj->toBaseContext();
        if (ctx != nullptr)
        {
            obj = ctx->getMechanicalState();
        }
        return obj;
    }
    else
    {
        msg_error("BaseObjectDescription") << "findObject: Node "<<nodeName<<" NOT FOUND.";
        return nullptr;
    }
}
} // namespace sofa::core::objectmodel
