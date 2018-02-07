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
            id = -(int)helper::TagFactory::getID(std::string(s.begin()+1,s.end()));
        }
        else
        {
            id = (int)helper::TagFactory::getID(s);
        }
    }
}

Tag::operator std::string() const
{
    if (id == 0) return std::string("0");
    else if (id > 0) return helper::TagFactory::getName((unsigned)id);
    else return std::string("!")+helper::TagFactory::getName((unsigned)-id);
}

bool TagSet::includes(const TagSet& t) const
{
    //return !empty() && std::includes( this->begin(), this->end(), t.begin(), t.end());
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
#if 1
    // Simple but not optimal version
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
#else
    // First test : no negative tag from t should appear as positive in this
    if (t.begin()->negative())
    {
        std::set<Tag>::const_reverse_iterator first1, last1;
        std::set<Tag>::const_iterator first2, last2;
        first1 = this->rbegin(); last1 = this->rend();
        first2 = t.begin(); last2 = t.end();
        for (; first2 != last2; ++first1)
        {
            if (first1 == last1) break; // no more tags in this
            Tag t1 = *first1;
            if (t1.negative()) break; // no more positive tags in this
            Tag t2 = *first2;
            if (!t2.negative()) break; // no more negative tags in t
            if (-t1 == t2)
                return false; // found an excluded tag
            if (!(-t1 < t2))
                ++first2;
        }
    }
    // Second test : all positive tags from t should appear as positive in this
    if (!t.rbegin()->negative())
    {
        std::set<Tag>::const_iterator first1, last1;
        std::set<Tag>::const_iterator first2, last2;
        first1 = this->lower_bound(Tag(0)); last1 = this->end();
        first2 = t.lower_bound(Tag(0)); last2 = t.end();
        //for(; first1 != last1 && first1->negative(); ++first1); // skip negative tags in this
        //for(; first2 != last2 && first2->negative(); ++first2); // skip negative tags in t
        for (; first2 != last2; ++first1)
        {
            if (first1 == last1)
                return false; // no more positive tags in this
            Tag t1 = *first1;
            Tag t2 = *first2;
            if (t2 < t1)
                return false; // tag not found
            if (!(t1 < t2))
                ++first2;
        }
    }
    return true; // all tests passed
#endif
}

} // namespace objectmodel

} // namespace core

} // namespace sofa
