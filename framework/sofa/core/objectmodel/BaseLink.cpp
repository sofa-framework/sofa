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
        std::string path = getLinkedName(i);
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

} // namespace objectmodel

} // namespace core

} // namespace sofa
