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
#include <set>

namespace sofa::core::objectmodel
{

class SOFA_CORE_API TagSet
{
public:
    using iterator = std::set<Tag>::iterator;
    using const_iterator = std::set<Tag>::const_iterator;
    using reverse_iterator = std::set<Tag>::reverse_iterator;
    using const_reverse_iterator = std::set<Tag>::const_reverse_iterator;
    using value_type = Tag;

    TagSet() = default;
    /// Automatic conversion between a tag and a tagset composed of this tag
    TagSet(const Tag& t);
    /// Returns true if this TagSet contains specified tag
    bool includes(const Tag& t) const;
    /// Returns true if this TagSet contains all specified tags
    bool includes(const TagSet& t) const;

    iterator find(const Tag& _Keyval);
    const_iterator find(const Tag& _Keyval) const;

    [[nodiscard]] bool empty() const noexcept;

    [[nodiscard]] std::size_t size() const noexcept;

    [[nodiscard]] std::size_t count(const Tag& _Keyval) const;

    [[nodiscard]] iterator begin() noexcept;

    [[nodiscard]] const_iterator begin() const noexcept;

    [[nodiscard]] iterator end() noexcept;

    [[nodiscard]] const_iterator end() const noexcept;

    [[nodiscard]] reverse_iterator rbegin() noexcept;

    [[nodiscard]] const_reverse_iterator rbegin() const noexcept;

    [[nodiscard]] reverse_iterator rend() noexcept;

    [[nodiscard]] const_reverse_iterator rend() const noexcept;

    [[nodiscard]] const_iterator cbegin() const noexcept;

    [[nodiscard]] const_iterator cend() const noexcept;

    [[nodiscard]] const_reverse_iterator crbegin() const noexcept;

    [[nodiscard]] const_reverse_iterator crend() const noexcept;

    std::pair<iterator, bool> insert(const value_type& _Val);

    iterator erase(const_iterator _Where) noexcept;
    iterator erase(const_iterator _First, const_iterator _Last) noexcept;
    std::size_t erase(const Tag& _Keyval) noexcept;

    void clear() noexcept;

private:
    std::set<Tag> m_set;
};

SOFA_CORE_API std::ostream &operator<<(std::ostream &o, const sofa::core::objectmodel::TagSet& tagSet);
SOFA_CORE_API std::istream &operator>>(std::istream &in, sofa::core::objectmodel::TagSet& tagSet);

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
