/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
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

#include <sstream>

namespace sofa
{

namespace core
{

namespace objectmodel
{

BaseLink::BaseLink(const BaseInitLink& init, LinkFlags flags)
    : m_flags(flags), m_name(init.name), m_help(init.help)
{
    //m_counters.assign(0);
    //m_isSets.assign(false);
}

BaseLink::~BaseLink()
{
}

/// Print the value of the associated variable
void BaseLink::printValue( std::ostream& o ) const
{
    unsigned int size = getSize();
    bool first = true;
    for (unsigned int i=0; i<size; ++i)
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

void BaseLink::copyAspect(int /*destAspect*/, int /*srcAspect*/)
{
    //m_counters[destAspect] = m_counters[srcAspect];
    //m_isSets[destAspect] = m_isSets[srcAspect];
}

void BaseLink::releaseAspect(int /*aspect*/)
{
}

bool BaseLink::ParseString(const std::string& text, std::string* path, std::string* data, Base* owner)
{
    if (text.empty())
    {
        if (owner) owner->serr << "ERROR parsing Link \""<<text<<"\": empty path." << owner->sendl;
        else std::cerr << "ERROR parsing Link \""<<text<<"\": empty path." << std::endl;
        return false;
    }
    if (text[0] != '@')
    {
        if (owner) owner->serr << "ERROR parsing Link \""<<text<<"\": first character should be '@'." << owner->sendl;
        else std::cerr << "ERROR parsing Link \""<<text<<"\": first character should be '@'." << std::endl;
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
        if (owner) owner->serr << "ERROR parsing Link \""<<text<<"\": a Data field name is specified while an object was expected." << owner->sendl;
        else std::cerr << "ERROR parsing Link \""<<text<<"\": a Data field name is specified while an object was expected." << std::endl;
        return false;
    }

    if (data && data->empty() && posDot == std::string::npos)
    {
        if (owner) owner->serr << "ERROR parsing Link \""<<text<<"\": a Data field name is required." << owner->sendl;
        else std::cerr << "ERROR parsing Link \""<<text<<"\": a Data field name is required." << std::endl;
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
            if (owner) owner->serr << "ERROR parsing Link \""<<text<<"\": missing closing bracket ']'." << owner->sendl;
            else std::cerr << "ERROR parsing Link \""<<text<<"\": missing closing bracket ']'." << std::endl;
            return false;
        }
        if ((*path)[0] == '[' && (*path)[1] != '-' && (*path)[1] != ']')
        {
            if (owner) owner->serr << "ERROR parsing Link \""<<text<<"\": bracket syntax can only be used for self-reference or preceding objects with a negative index." << owner->sendl;
            else std::cerr << "ERROR parsing Link \""<<text<<"\": bracket syntax can only be used for self-reference or preceding objects with a negative index." << std::endl;
            return false;
        }
    }
    //std::cout << "LINK: Parsed \"" << text << "\":";
    //if (path) std::cout << " path=\"" << *path << "\"";
    //if (data) std::cout << " data=\"" << *data << "\"";
    //std::cout << std::endl;
    return true;
}

std::string BaseLink::CreateString(const std::string& path, const std::string& data)
{
    std::string result = "@";
    if (!path.empty()) result += path;
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
    BaseObject* o = dynamic_cast<BaseObject*>(dest);
    BaseObject* f = dynamic_cast<BaseObject*>(from);
    BaseContext* ctx = dynamic_cast<BaseContext*>(from);
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
        BaseNode* n = dynamic_cast<BaseNode*>(o->getContext());
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
        BaseNode* n = dynamic_cast<BaseNode*>(dest);
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

#ifndef SOFA_DEPRECATE_OLD_API
std::string BaseLink::ConvertOldPath(const std::string& path, const char* oldName, const char* newName, Base* obj, bool showWarning)
{
    std::string newPath;
    if (path.empty())
        newPath = std::string();
    else if (path[0] == '@')
        newPath = path; // this is actually a path with the current syntax
    else if (path[0] == '/')
        newPath = std::string("@") + path; // absolute path
    else if (path == "..")
        newPath = std::string("@./"); // special case: current context
    else if (path == ".")
        newPath = std::string("@[]"); // special case: current object
    else if (path.substr(0,3) == std::string("../"))
        newPath = std::string("@") + path.substr(3); // remove one parent level
    else
        newPath = std::string("@") + path; // path from one of the parent nodes
    /*
        {
            if (obj && oldName && newName)
                obj->serr << "Invalid and deprecated path "<< oldName << "=\"" << path << "\". Replace it with a path specified as " << newName << " and using the new '@' prefixed syntax." << obj->sendl;
            else if (obj)
                obj->serr << "Invalid and deprecated path \"" << path << "\". Replace it with a path using the new '@' prefixed syntax." << obj->sendl;
            else if (oldName && newName)
                std::cerr << "Invalid and deprecated path "<< oldName << "=\"" << path << "\". Replace it with a path specified as " << newName << " and using the new '@' prefixed syntax." << std::endl;
            else
                std::cerr << "Invalid and deprecated path \"" << path << "\". Replace it with a path using the new '@' prefixed syntax." << std::endl;
            return path;
        }
    */
    if (obj && showWarning)
    {
        if (oldName && newName)
            obj->sout << "Deprecated syntax "<< oldName << "=\"" << path << "\". Replace with " << newName << "=\"" << newPath << "\"." << obj->sendl;
        else
            obj->sout << "Deprecated syntax \"" << path << "\". Replace with \"" << newPath << "\"." << obj->sendl;
    }
    return newPath;
}
#endif

} // namespace objectmodel

} // namespace core

} // namespace sofa
