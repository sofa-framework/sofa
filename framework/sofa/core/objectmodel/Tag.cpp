/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/helper/TagFactory.h>
#include <sofa/core/objectmodel/Tag.h>
#include <algorithm> // for std::includes

namespace sofa
{

namespace core
{

namespace objectmodel
{

Tag::Tag(const std::string& s)
    : id(0)
{
    if (!s.empty())
    {
        id = helper::TagFactory::getID(s);
    }
}

Tag::operator std::string() const
{
    if (id == 0) return std::string("0");
    else return helper::TagFactory::getName(id);
}

bool TagSet::includes(const TagSet& t) const
{
    return !empty() && std::includes( this->begin(), this->end(), t.begin(), t.end());
}

} // namespace objectmodel

} // namespace core

} // namespace sofa
