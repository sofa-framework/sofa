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
#include <cstring>
#include <queue>
#include <difflib.h>
#include <sofa/helper/DiffLib.h>

namespace sofa::helper
{

std::vector<std::tuple<std::string, SReal>> SOFA_HELPER_API getClosestMatch(const std::string& needle,
                                                                            const std::vector<std::string>& haystack,
                                                                            const Size numEntries, const SReal thresold)
{
    class Tuple
    {
    public:
        Tuple(float ratio_, std::string value_)
        {
            ratio = ratio_;
            value = value_;
        }
        float ratio;
        std::string value;
    };
    auto cmp = [](Tuple& left, Tuple& right) { return left.ratio < right.ratio; };
    std::priority_queue<Tuple, std::vector<Tuple>, decltype(cmp)> q3(cmp);

    for(auto& s : haystack)
    {
        auto foo = difflib::MakeSequenceMatcher(needle,s);
        q3.push(Tuple(foo.ratio(), s));
    }
    std::vector<std::tuple<std::string, SReal>> result;
    while(!q3.empty() && result.size()<=numEntries)
    {
        if(q3.top().ratio < thresold)
            break;
        result.push_back(std::make_tuple(q3.top().value, q3.top().ratio));
        q3.pop();
    }
    return result;
};

} // namespace sofa

