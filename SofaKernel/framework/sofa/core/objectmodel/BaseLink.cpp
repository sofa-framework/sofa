/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/core/objectmodel/BaseLink.h>
#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/logging/Messaging.h>

#include <sstream>

namespace sofa
{

namespace core
{

namespace objectmodel
{

BaseLink::BaseLink(LinkFlags flags)
    : m_flags(flags)
{
    m_counters.assign(0);
    //m_isSets.assign(false);
}

BaseLink::BaseLink(const BaseInitLink& init, LinkFlags flags)
    : m_flags(flags), m_name(init.name), m_help(init.help)
{
    m_counters.assign(0);
    //m_isSets.assign(false);
}

BaseLink::~BaseLink()
{
}

/// Print the value of the associated variable
void BaseLink::printValue( std::ostream& o ) const
{
    std::size_t size = getSize();
    bool first = true;
    for (std::size_t i = 0; i<size; ++i)
    {
        std::string path = getLinkedPath(i);
        if (path.empty()) continue;
        if (first) first = false;
        else o << ' ';
        o << path;
    }
}

/// Print the value of the associated variable
std::string BaseLink::getValueString() const
{
    std::ostringstream o;
    printValue(o);
    return o.str();
}

/// Print the value type of the associated variable
std::string BaseLink::getValueTypeString() const
{
    const BaseClass* c = getDestClass();
    if (!c) return "void";
    std::string t = c->className;
    if (!c->templateName.empty())
    {
        t += '<';
        t += c->templateName;
        t += '>';
    }
    return t;
}

/// Copy the value of an aspect into another one.
void BaseLink::copyAspect(int destAspect, int srcAspect)
{
    m_counters[destAspect] = m_counters[srcAspect];
    //m_isSets[destAspect] = m_isSets[srcAspect];
}

/// Release memory allocated for the specified aspect.
void BaseLink::releaseAspect(int aspect)
{
    m_counters[aspect] = -1;
}

bool BaseLink::ParseString(const std::string& text, std::string* path, std::string* data, Base* owner)
{
    if (text.empty())
    {
        if (owner) msg_error(owner) << "Parsing Link \""<<text<<"\": empty path.";
        else msg_error("BaseLink") << "ParseString: \""<<text<<"\": empty path.";
        return false;
    }
    if (text[0] != '@')
    {
        if (owner) msg_error(owner) << "ERROR parsing Link \""<<text<<"\": first character should be '@'.";
        else msg_error("BaseLink") << "ParseString: \""<<text<<"\": first character should be '@'.";
        return false;
    }
    std::size_t posPath = text.rfind('/');
    if (posPath == std::string::npos) posPath = 0;
    std::size_t posDot = text.rfind('.');
    if (posDot < posPath) posDot = std::string::npos; // dots can appear within the path
    if (posDot == text.size()-1 && text[posDot-1] == '.') posDot = std::string::npos; // double dots can appear at the end of the path
    if (posDot == text.size()-1 && (text[posDot-1] == '/' || posDot == 1)) posDot = std::string::npos; // a single dot is allowed at the end of the path, although it is better to end it with '/' instead in order to remove any ambiguity
    if (!data && posDot != std::string::npos)
    {
        if (owner) msg_error(owner) << "Parsing Link \""<<text<<"\": a Data field name is specified while an object was expected.";
        else msg_error("BaseLink") << "ParseString: \""<<text<<"\": a Data field name is specified while an object was expected.";
        return false;
    }

    if (data && data->empty() && posDot == std::string::npos)
    {
        if (owner) msg_error(owner) << "Parsing Link \""<<text<<"\": a Data field name is required." ;
        else msg_error("BaseLink") << "ParseString: \""<<text<<"\": a Data field name is required.";
        return false;
    }

    if (!data || posDot == std::string::npos)
    {
        if (path)
            *path = text.substr(1);
    }
    else
    {
        if (path)
            *path = text.substr(1,posDot-1);
        *data = text.substr(posDot+1);
    }
    if (path && !path->empty())
    {
        if ((*path)[0] == '[' && (*path)[path->size()-1] != ']')
        {
            if (owner) msg_error(owner) << "Parsing Link \""<<text<<"\": missing closing bracket ']'." ;
            else msg_error("BaseLink") << "ParseString: \""<<text<<"\": missing closing bracket ']'.";
            return false;
        }
        if ((*path)[0] == '[' && (*path)[1] != '-' && (*path)[1] != ']')
        {
            if (owner) msg_error(owner) << "Parsing Link \""<<text<<"\": bracket syntax can only be used for self-reference or preceding objects with a negative index." << owner->sendl;
            else msg_error("BaseLink") << "ParseString: \""<<text<<"\": bracket syntax can only be used for self-reference or preceding objects with a negative index.";
            return false;
        }
    }
    return true;
}

std::string BaseLink::CreateString(const std::string& path, const std::string& data)
{
    std::string result = "@";
    if (!path.empty()) result += path;
    else result = "@/" ;

    if (!data.empty())
    {
        if (result[result.size()-1] == '.')
            result += '/'; // path ends at a node designed with '.' or '..', so add '/' in order to separate it from the data part
        result += '.';
        result += data;
    }
    return result;
}

std::string BaseLink::CreateStringPath(Base* dest, Base* from)
{
    if (!dest || dest == from) return std::string("[]");
    BaseObject* o = dest->toBaseObject();
    BaseObject* f = from->toBaseObject();
    BaseContext* ctx = from->toBaseContext();
    if (!ctx && f) ctx = f->getContext();
    if (o)
    {
        std::string objectPath = o->getName();
        BaseObject* master = o->getMaster();
        while (master)
        {
            objectPath = master->getName() + std::string("/") + objectPath;
            master = master->getMaster();
        }
        BaseNode* n = o->getContext()->toBaseNode();
        if (f && o->getContext() == ctx)
            return objectPath;
        else if (n)
            return n->getPathName() + std::string("/") + objectPath; // TODO: compute relative path
        else
            return objectPath; // we could not determine destination path, specifying simply its name might be enough to find it back
    }
    else // dest is a context
    {
        if (f && ctx == dest)
            return std::string("./");
        BaseNode* n = dest->toBaseNode();
        if (n) return n->getPathName(); // TODO: compute relative path
        else return dest->getName(); // we could not determine destination path, specifying simply its name might be enough to find it back
    }
}

std::string BaseLink::CreateStringData(BaseData* data)
{
    if (!data) return std::string();
    return data->getName();
}
std::string BaseLink::CreateString(Base* object, Base* from)
{
    return CreateString(CreateStringPath(object,from));
}
std::string BaseLink::CreateString(BaseData* data, Base* from)
{
    return CreateString(CreateStringPath(data->getOwner(),from),CreateStringData(data));
}
std::string BaseLink::CreateString(Base* object, BaseData* data, Base* from)
{
    return CreateString(CreateStringPath(object,from),CreateStringData(data));
}

} // namespace objectmodel

} // namespace core

} // namespace sofa
