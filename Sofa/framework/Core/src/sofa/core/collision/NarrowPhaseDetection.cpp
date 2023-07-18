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
#include <sofa/core/visual/VisualParams.h>

namespace sofa::core::collision
{

NarrowPhaseDetection::~NarrowPhaseDetection()
{
    for (const auto& it : m_outputsMap)
    {
        DetectionOutputVector* do_vec = it.second;

        if (do_vec != nullptr)
        {
            do_vec->clear();
            do_vec->release();
        }
    }
}

void NarrowPhaseDetection::beginNarrowPhase()
{
    for (DetectionOutputMap::iterator it = m_outputsMap.begin(); it != m_outputsMap.end(); it++)
    {
        DetectionOutputVector *do_vec = (it->second);

        if (do_vec != nullptr)
            do_vec->clear();
    }
}

void NarrowPhaseDetection::draw(const core::visual::VisualParams* vparams)
{
    if(! vparams->displayFlags().getShowDetectionOutputs()) return;

    std::vector<type::Vec3> points;

    for (auto mapIt = m_outputsMap.begin(); mapIt!=m_outputsMap.end() ; ++mapIt)
    {
        for (unsigned idx = 0; idx != (*mapIt).second->size(); ++idx)
        {
            points.push_back((*mapIt).second->getFirstPosition(idx));
            points.push_back((*mapIt).second->getSecondPosition(idx));
        }
    }
    vparams->drawTool()->drawLines(points,5, type::g_red);
    vparams->drawTool()->drawPoints(points, 10, type::g_blue);

}

void NarrowPhaseDetection::addCollisionPairs(const sofa::type::vector< std::pair<core::CollisionModel*, core::CollisionModel*> >& v)
{
    for (sofa::type::vector< std::pair<core::CollisionModel*, core::CollisionModel*> >::const_iterator it = v.begin(); it!=v.end(); it++)
        addCollisionPair(*it);

    // m_outputsMap should just be filled in addCollisionPair function
    m_primitiveTestCount = m_outputsMap.size();
}

void NarrowPhaseDetection::endNarrowPhase()
{
    for (auto it = m_outputsMap.begin(); it != m_outputsMap.end();)
    {
        DetectionOutputVector *do_vec = (it->second);
        if (!do_vec || do_vec->empty())
        {
            if (do_vec)
            {
                do_vec->release();
            }
            m_outputsMap.erase(it++);
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
    const auto res = m_outputsMap.insert(m_outputsMap.end(), {cm_pair, nullptr});
    return res->second;
}

void NarrowPhaseDetection::changeInstanceNP(Instance inst)
{
    m_storedOutputsMap[instance].swap(m_outputsMap);
    m_outputsMap.swap(m_storedOutputsMap[inst]);
}

} // namespace sofa::core::collision
