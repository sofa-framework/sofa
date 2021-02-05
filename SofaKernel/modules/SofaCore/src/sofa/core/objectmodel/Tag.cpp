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
#include <sofa/helper/TagFactory.h>
#include <sofa/core/objectmodel/Tag.h>
//#include <algorithm> // for std::includes

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
        if (s[0] == '!')
        {
            id = -int(helper::TagFactory::getID(std::string(s.begin()+1,s.end())));
        }
        else
        {
            id = int(helper::TagFactory::getID(s));
        }
    }
}

Tag::operator std::string() const
{
    if (id == 0) return std::string("0");
    else if (id > 0) return helper::TagFactory::getName(unsigned(id));
    else return std::string("!")+helper::TagFactory::getName(unsigned(-id));
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

std::ostream& operator<<(std::ostream& o, const Tag& t)
{
    return o << std::string(t);
}

std::istream& operator>>(std::istream& i, Tag& t)
{
    std::string s;
    i >> s;
    t = Tag(s);
    return i;
}


} // namespace objectmodel

} // namespace core

} // namespace sofa
