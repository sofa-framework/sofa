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
#include <sofa/defaulttype/typeinfo/TypeInfo_Text.h>

namespace sofa::core::objectmodel
{

/**
 *  \brief A Tag is a string (internally converted to an integer), attached to objects in order to define subsets to process by specific visitors.
 *
 */
class SOFA_CORE_API Tag
{
public:

    Tag() : id(0) {}

    /// A tag is constructed from a string and appears like one after, without actually storing a string
    Tag(const std::string& s);

    /// This constructor should be used only if really necessary
    explicit Tag(int idtag) : id(idtag) {}

    /// Any operation requiring a string can be used on a tag using this conversion
    operator std::string() const;

    bool operator==(const Tag& t) const { return id == t.id; }
    bool operator!=(const Tag& t) const { return id != t.id; }
    bool operator<(const Tag& t) const { return id < t.id; }
    bool operator>(const Tag& t) const { return id > t.id; }
    bool operator<=(const Tag& t) const { return id <= t.id; }
    bool operator>=(const Tag& t) const { return id >= t.id; }
    bool operator!() const { return !id; }

    bool negative() const { return id < 0; }
    Tag operator-() const { return Tag(-id); }

    SOFA_CORE_API friend std::ostream& operator<<(std::ostream& o, const Tag& t);
    SOFA_CORE_API friend std::istream& operator>>(std::istream& i, Tag& t);

protected:
    int id;
};

} //namespace sofa::core::objectmodel

// Specialization of the defaulttype::DataTypeInfo type traits template
namespace sofa::defaulttype
{

template<>
struct DataTypeInfo< sofa::core::objectmodel::Tag > : public TextTypeInfo<sofa::core::objectmodel::Tag >
{
    static const char* name() { return "Tag"; }
};

} //namespace sofa::defaulttype
