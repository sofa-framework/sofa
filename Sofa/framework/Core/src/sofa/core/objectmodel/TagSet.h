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
#pragma once

#include <sofa/core/config.h>
#include <sofa/core/objectmodel/Tag.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Set.h>
#include <sofa/helper/set.h>

namespace sofa::core::objectmodel
{

class SOFA_CORE_API TagSet : public std::set<Tag>
{
public:
    TagSet() = default;
    /// Automatic conversion between a tag and a tagset composed of this tag
    TagSet(const Tag& t);
    /// Returns true if this TagSet contains specified tag
    bool includes(const Tag& t) const;
    /// Returns true if this TagSet contains all specified tags
    bool includes(const TagSet& t) const;
};

} // namespace sofa::core::objectmodel

// Specialization of the defaulttype::DataTypeInfo type traits template
namespace sofa::defaulttype
{

template<>
struct DataTypeInfo< sofa::core::objectmodel::TagSet > : public SetTypeInfo<sofa::core::objectmodel::TagSet >
{
    static const char* name() { return "TagSet"; }
};

} // namespace sofa::defaulttype
