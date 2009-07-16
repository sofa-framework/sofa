/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <sofa/component/collision/PointLocalMinDistanceFilter.h>

#include <sofa/component/collision/LineModel.h>
#include <sofa/component/collision/LocalMinDistanceFilter.inl>
#include <sofa/component/topology/PointData.inl>

#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <sofa/core/componentmodel/topology/Topology.h>

#include <sofa/simulation/common/Node.h>

#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace collision
{


void PointInfo::buildFilter(const Point &p)
{
    using sofa::simulation::Node;
    using sofa::helper::vector;
    using sofa::core::componentmodel::topology::BaseMeshTopology;

    if (isRigid())
    {
        // update rigid
    }
    else
    {
        const Vector3 &pt = p.p();

        Node* node = dynamic_cast< Node* >(p.getCollisionModel()->getContext());
        if ( !(node->get< LineModel >()) )
        {
            m_noLineModel = true;
            return;
        }

        BaseMeshTopology* topology = p.getCollisionModel()->getMeshTopology();
        vector< Vector3 >& x = *(p.getCollisionModel()->getMechanicalState()->getX());

        const vector< unsigned int >& trianglesAroundVertex = topology->getTrianglesAroundVertex(p.getIndex());

        vector< unsigned int >::const_iterator triIt = trianglesAroundVertex.begin();
        vector< unsigned int >::const_iterator triItEnd = trianglesAroundVertex.end();

        Vector3 nMean;

        while (triIt != triItEnd)
        {
            const BaseMeshTopology::Triangle& triangle = topology->getTriangle(*triIt);

            Vector3 nCur = (x[triangle[1]] - x[triangle[0]]).cross(x[triangle[2]] - x[triangle[0]]);
            nCur.normalize();
            nMean += nCur;

            ++triIt;
        }

        const vector< unsigned int >& edgesAroundVertex = topology->getEdgesAroundVertex(p.getIndex());

        if (trianglesAroundVertex.empty())
        {
            vector< unsigned int >::const_iterator edgeIt = edgesAroundVertex.begin();
            vector< unsigned int >::const_iterator edgeItEnd = edgesAroundVertex.end();

            while (edgeIt != edgeItEnd)
            {
                const BaseMeshTopology::Edge& edge = topology->getEdge(*edgeIt);

                Vector3 l = (pt - x[edge[0]]) + (pt - x[edge[1]]);
                l.normalize();
                nMean += l;

                ++edgeIt;
            }
        }

        if (nMean.norm() > 0.0000000001)
            nMean.normalize();
        else
            std::cerr << "WARNING PointInfo m_nMean is null" << std::endl;

        vector< unsigned int >::const_iterator edgeIt = edgesAroundVertex.begin();
        vector< unsigned int >::const_iterator edgeItEnd = edgesAroundVertex.end();

        while (edgeIt != edgeItEnd)
        {
            const BaseMeshTopology::Edge& edge = topology->getEdge(*edgeIt);

            Vector3 l = (pt - x[edge[0]]) + (pt - x[edge[1]]);
            l.normalize();

            double computedAngleCone = dot(nMean , l) * m_lmdFilters->getConeExtension();

            if (computedAngleCone < 0)
                computedAngleCone = 0.0;

            computedAngleCone += m_lmdFilters->getConeMinAngle();

            m_computedData.push_back(std::make_pair(l, computedAngleCone));

            ++edgeIt;
        }
    }

    setValid();
}



bool PointInfo::validate(const Point &p, const defaulttype::Vector3 &PQ)
{
    if (isValid())
    {
        if (m_noLineModel)
            return true;

        TDataContainer::const_iterator it = m_computedData.begin();
        TDataContainer::const_iterator itEnd = m_computedData.end();

        while (it != itEnd)
        {
            if (dot(it->first , PQ) < (-it->second * PQ.norm()))
                return false;

            ++it;
        }

        return true;
    }
    else
    {
        buildFilter(p);
        return validate(p, PQ);
    }
}



void PointLocalMinDistanceFilter::init()
{
    core::componentmodel::topology::BaseMeshTopology *bmt = getContext()->getMeshTopology();

    if (bmt != 0)
    {
        helper::vector< PointInfo >& pInfo = *(m_pointInfo.beginEdit());
        pInfo.resize(bmt->getNbPoints());
        m_pointInfo.endEdit();
    }
}



void PointLocalMinDistanceFilter::handleTopologyChange()
{
    core::componentmodel::topology::BaseMeshTopology *bmt = getContext()->getMeshTopology();

    assert(bmt != 0);

    std::list< const core::componentmodel::topology::TopologyChange * >::const_iterator itBegin = bmt->firstChange();
    std::list< const core::componentmodel::topology::TopologyChange * >::const_iterator itEnd = bmt->lastChange();

    m_pointInfo.handleTopologyEvents(itBegin, itEnd);
}



void PointLocalMinDistanceFilter::LMDFilterPointCreationFunction(int /*pointIndex*/, void* param, PointInfo &pInfo, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >&)
{
    const PointLocalMinDistanceFilter *pLMDFilter = static_cast< const PointLocalMinDistanceFilter * >(param);
    pInfo.setLMDFilters(pLMDFilter);
}



SOFA_DECL_CLASS(PointLocalMinDistanceFilter)

int PointLocalMinDistanceFilterClass = core::RegisterObject("This class manages Point collision models cones filters computations and updates.")
        .add< PointLocalMinDistanceFilter >()
        ;

} // namespace collision

} // namespace component

} // namespace sofa
