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
#include <sofa/simulation/MappingGraph.h>

namespace sofa::simulation
{

namespace
{
template <class T>
bool haveCommonElement(const std::vector<T>& a, const std::vector<T>& b)
{
    if (&a == &b) return true;

    const std::vector<T>& smaller = (a.size() < b.size()) ? a : b;
    const std::vector<T>& larger = (a.size() < b.size()) ? b : a;

    std::unordered_set<T> elements(smaller.begin(), smaller.end());

    return std::any_of(larger.begin(), larger.end(),
                       [&elements](const T& val) { return elements.count(val); });
}
}

void findIndependentMappings(
    const std::vector<core::BaseMapping*>& allMappings,
    std::queue<core::BaseMapping*>& independentMappings,
    MappingGraphDirection direction)
{
    for (auto* mapping1 : allMappings)
    {
        type::vector<core::BaseState*> inputs1 = mapping1->getFrom();
        type::vector<core::BaseState*> outputs1 = mapping1->getTo();
        bool isIndependent = true;
        for (auto* mapping2 : allMappings)
        {
            if (mapping1 != mapping2)
            {
                auto inputs2 = mapping2->getFrom();
                auto outputs2 = mapping2->getTo();
                const bool commonElement =
                    (direction == MappingGraphDirection::FORWARD) ?
                        haveCommonElement(inputs1, outputs2) :
                        haveCommonElement(outputs1, inputs2);
                if (commonElement)
                {
                    // mapping1 is not an independent mapping because it shares common elements with mapping2
                    isIndependent = false;
                    break;
                }
                if (haveCommonElement(outputs1, outputs2))
                {
                    msg_warning("MappingGraph")
                        << "Mapping " << mapping1->getName() << " has common output with "
                        << mapping2->getName();
                }
            }
        }

        if (isIndependent)
        {
            independentMappings.push(mapping1);
        }
    }
}

}  // namespace sofa::simulation
