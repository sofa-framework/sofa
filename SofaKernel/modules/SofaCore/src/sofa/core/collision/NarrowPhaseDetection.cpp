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
#include <sofa/core/collision/NarrowPhaseDetection.h>

namespace sofa::core::collision
{

void NarrowPhaseDetection::beginNarrowPhase()
{
    for (DetectionOutputMap::iterator it = m_outputsMap.begin(); it != m_outputsMap.end(); it++)
    {
        DetectionOutputVector *do_vec = (it->second);

        if (do_vec != nullptr)
            do_vec->clear();
    }
}

void NarrowPhaseDetection::addCollisionPairs(const sofa::helper::vector< std::pair<core::CollisionModel*, core::CollisionModel*> >& v)
{
    for (sofa::helper::vector< std::pair<core::CollisionModel*, core::CollisionModel*> >::const_iterator it = v.begin(); it!=v.end(); it++)
        addCollisionPair(*it);

    // m_outputsMap should just be filled in addCollisionPair function
    m_primitiveTestCount = m_outputsMap.size();
}

void NarrowPhaseDetection::endNarrowPhase()
{
    DetectionOutputMap::iterator it = m_outputsMap.begin();

    while (it != m_outputsMap.end())
    {
        DetectionOutputVector *do_vec = (it->second);

        if (!do_vec || do_vec->empty())
        {
            /// @todo Optimization
            DetectionOutputMap::iterator iterase = it;
            ++it;
            m_outputsMap.erase(iterase);
            if (do_vec) do_vec->release();
        }
        else
        {
            ++it;
        }
    }
}

size_t NarrowPhaseDetection::getPrimitiveTestCount() const
{
    return m_primitiveTestCount;
}

auto NarrowPhaseDetection::getDetectionOutputs() const -> const DetectionOutputMap&
{
    return m_outputsMap;
}

DetectionOutputVector*& NarrowPhaseDetection::getDetectionOutputs(CollisionModel *cm1, CollisionModel *cm2)
{
    std::pair< CollisionModel*, CollisionModel* > cm_pair = std::make_pair(cm1, cm2);

    DetectionOutputMap::iterator it = m_outputsMap.find(cm_pair);

    if (it == m_outputsMap.end())
    {
        // new contact
        it = m_outputsMap.insert( std::make_pair(cm_pair, static_cast< DetectionOutputVector * >(0)) ).first;
    }

    return it->second;
}

void NarrowPhaseDetection::changeInstanceNP(Instance inst)
{
    m_storedOutputsMap[instance].swap(m_outputsMap);
    m_outputsMap.swap(m_storedOutputsMap[inst]);
}

} // namespace sofa::core::collision
