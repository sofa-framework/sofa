/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
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
        //std::cout << "Found node "<<nodeName<<": "<<node->getName()<<std::endl;
        Base* obj = node->getObject();
        BaseContext* ctx = dynamic_cast<BaseContext*>(obj);
        if (ctx != NULL)
        {
            //std::cout << "Node "<<nodeName<<" is a context, returning MechanicalState."<<std::endl;
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
