/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_OBJECTMODEL_TAG_H
#define SOFA_CORE_OBJECTMODEL_TAG_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/helper/set.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/core/core.h>
#include <iostream>
#include <string>

namespace sofa
{

namespace core
{

namespace objectmodel
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
    explicit Tag(int id) : id(id) {}

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

    friend std::ostream& operator<<(std::ostream& o, const Tag& t)
    {
        return o << (std::string)t;
    }

    friend std::istream& operator>>(std::istream& i, Tag& t)
    {
        std::string s;
        i >> s;
        t = Tag(s);
        return i;
    }

protected:
    int id;
};

class SOFA_CORE_API TagSet : public std::set<Tag>
{
public:
    TagSet() {}
    /// Automatic conversion between a tag and a tagset composed of this tag
    TagSet(const Tag& t) { this->insert(t); }
    /// Returns true if this TagSet contains specified tag
    bool includes(const Tag& t) const { return this->count(t) > 0; }
    /// Returns true if this TagSet contains all specified tags
    bool includes(const TagSet& t) const;
};

} // namespace objectmodel

} // namespace core

// Specialization of the defaulttype::DataTypeInfo type traits template

namespace defaulttype
{

template<>
struct DataTypeInfo< sofa::core::objectmodel::Tag > : public TextTypeInfo<sofa::core::objectmodel::Tag >
{
    static const char* name() { return "Tag"; }
};

template<>
struct DataTypeInfo< sofa::core::objectmodel::TagSet > : public SetTypeInfo<sofa::core::objectmodel::TagSet >
{
    static const char* name() { return "TagSet"; }
};

} // namespace defaulttype

} // namespace sofa

#endif
