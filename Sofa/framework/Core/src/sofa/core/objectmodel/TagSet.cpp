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
#include <sofa/core/objectmodel/TagSet.h>
#include <sofa/helper/StringUtils.h>

namespace sofa::core::objectmodel
{

TagSet::TagSet(const Tag& t)
{
    m_set.insert(t);
}

bool TagSet::includes(const Tag& t) const
{
    return m_set.count(t) > 0;
}

bool TagSet::includes(const TagSet& t) const
{
    if (t.empty())
        return true;
    if (empty())
    {
        // An empty TagSet satisfies the conditions only if either :
        // t is also empty (already handled)
        // t only includes negative tags
        if (*t.rbegin() <= Tag(0))
            return true;
        // t includes the "0" tag
        if (t.count(Tag(0)) > 0)
            return true;
        // otherwise the TagSet t does not "include" empty sets
        return false;
    }
    for (std::set<Tag>::const_iterator first2 = t.begin(), last2 = t.end();
        first2 != last2; ++first2)
    {
        Tag t2 = *first2;
        if (t2 == Tag(0)) continue; // tag "0" is used to indicate that we should include objects without any tag
        if (!t2.negative())
        {
            if (this->count(t2) == 0)
                return false; // tag not found in this
        }
        else
        {
            if (this->count(-t2) > 0)
                return false; // tag found in this
        }
    }
    return true;
}

TagSet::iterator TagSet::find(const Tag& _Keyval)
{
    return m_set.find(_Keyval);
}

TagSet::const_iterator TagSet::find(const Tag& _Keyval) const
{
    return m_set.find(_Keyval);
}

bool TagSet::empty() const noexcept
{
    return m_set.empty();
}

std::size_t TagSet::size() const noexcept
{
    return m_set.size();
}

std::size_t TagSet::count(const Tag& _Keyval) const
{
    return m_set.count(_Keyval);
}

TagSet::iterator TagSet::begin() noexcept
{
    return m_set.begin();
}

TagSet::const_iterator TagSet::begin() const noexcept
{
    return m_set.begin();
}

TagSet::iterator TagSet::end() noexcept
{
    return m_set.end();
}

TagSet::const_iterator TagSet::end() const noexcept
{
    return m_set.end();
}

TagSet::reverse_iterator TagSet::rbegin() noexcept
{
    return m_set.rbegin();
}

TagSet::const_reverse_iterator TagSet::rbegin() const noexcept
{
    return m_set.rbegin();
}

TagSet::reverse_iterator TagSet::rend() noexcept
{
    return m_set.rend();
}

TagSet::const_reverse_iterator TagSet::rend() const noexcept
{
    return m_set.rend();
}

TagSet::const_iterator TagSet::cbegin() const noexcept
{
    return m_set.cbegin();
}

TagSet::const_iterator TagSet::cend() const noexcept
{
    return m_set.cend();
}

TagSet::const_reverse_iterator TagSet::crbegin() const noexcept
{
    return m_set.crbegin();
}

TagSet::const_reverse_iterator TagSet::crend() const noexcept
{
    return m_set.crend();
}

std::pair<TagSet::iterator, bool> TagSet::insert(const value_type& _Val)
{
    return m_set.insert(_Val);
}

TagSet::iterator TagSet::erase(const_iterator _Where) noexcept
{
    return m_set.erase(_Where);
}

TagSet::iterator TagSet::erase(const_iterator _First,
    const_iterator _Last) noexcept
{
    return m_set.erase(_First, _Last);
}

std::size_t TagSet::erase(const Tag& _Keyval) noexcept
{
    return m_set.erase(_Keyval);
}

void TagSet::clear() noexcept
{
    m_set.clear();
}

std::ostream& operator<<(std::ostream& o,
    const sofa::core::objectmodel::TagSet& tagSet)
{
    o << sofa::helper::join(tagSet.begin(), tagSet.end(), ' ');
    return o;
}

std::istream& operator>>(std::istream& in,
    sofa::core::objectmodel::TagSet& tagSet)
{
    Tag t;
    tagSet.clear();
    while(in>>t)
        tagSet.insert(t);
    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
    return in;
}


} //namespace sofa::core::objectmodel

