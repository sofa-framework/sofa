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

#include <sofa/component/collision/LineLocalMinDistanceFilter.h>

#include <sofa/component/collision/LocalMinDistanceFilter.inl>
#include <sofa/component/topology/EdgeData.inl>
#include <sofa/component/topology/PointData.inl>

#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>

#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace collision
{


void LineInfo::buildFilter(const Line &l)
{
    using sofa::helper::vector;
    using sofa::core::componentmodel::topology::BaseMeshTopology;

    if (isRigid())
    {
        // update rigid
    }
    else
    {
        const Vector3 &pt1 = l.p1();
        const Vector3 &pt2 = l.p2();

        m_lineVector = pt2 - pt1;
        m_lineVector.normalize();

        BaseMeshTopology* topology = l.getCollisionModel()->getMeshTopology();

        vector< Vector3 >& x = *(l.getCollisionModel()->getMechanicalState()->getX());

        const sofa::helper::vector<unsigned int>& trianglesAroundEdge = topology->getTrianglesAroundEdge(l.getIndex());

        // filter if there are two triangles around the edge
        if (trianglesAroundEdge.size() != 2)
        {
            m_twoTrianglesAroundEdge = false;
        }

        // compute the normal of the triangle situated on the right
        const BaseMeshTopology::Triangle& triangleRight = topology->getTriangle(trianglesAroundEdge[0]);
        Vector3 n1 = cross(x[triangleRight[1]] - x[triangleRight[0]], x[triangleRight[2]] - x[triangleRight[0]]);
        n1.normalize();
        m_nMean = n1;
        m_triangleRight = -cross(n1, m_lineVector);
        m_triangleRight.normalize(); // necessary ?

        // compute the normal of the triangle situated on the left
        const BaseMeshTopology::Triangle& triangleLeft = topology->getTriangle(trianglesAroundEdge[1]);
        Vector3 n2 = cross(x[triangleLeft[1]] - x[triangleLeft[0]], x[triangleLeft[2]] - x[triangleLeft[0]]);
        n2.normalize();
        m_nMean += n2;
        m_triangleLeft = -cross(m_lineVector, n2);
        m_triangleLeft.normalize(); // necessary ?

        m_nMean.normalize();

        // compute the angle for the cone to filter contacts using the normal of the triangle situated on the right
        m_computedRightAngleCone = (m_nMean * m_triangleRight) * m_lmdFilters->getConeExtension();
        if (m_computedRightAngleCone < 0)
        {
            m_computedRightAngleCone = 0.0;
        }
        m_computedRightAngleCone += m_lmdFilters->getConeMinAngle();

        // compute the angle for the cone to filter contacts using the normal of the triangle situated on the left
        m_computedLeftAngleCone = (m_nMean * m_triangleLeft) * m_lmdFilters->getConeExtension();
        if (m_computedLeftAngleCone < 0)
        {
            m_computedLeftAngleCone = 0.0;
        }
        m_computedLeftAngleCone += m_lmdFilters->getConeMinAngle();
    }

    setValid();
}



bool LineInfo::validate(const Line &l, const defaulttype::Vector3 &PQ)
{
    if (isValid())
    {
        if (m_twoTrianglesAroundEdge)
        {
            if ((m_nMean * PQ) < 0)
                return false;

            if (m_triangleRight * PQ < -m_computedRightAngleCone * PQ.norm())
                return false;

            if (m_triangleLeft * PQ < -m_computedLeftAngleCone * PQ.norm())
                return false;
        }
        else
        {
            Vector3 PQnormalized = PQ;
            PQnormalized.normalize();

            if (fabs(dot(m_lineVector, PQnormalized)) > m_lmdFilters->getConeMinAngle() + 0.001)		// auto-collision case between
            {
                return false;
            }
        }

        return true;
    }
    else
    {
        buildFilter(l);
        return validate(l, PQ);
    }
}



void LineLocalMinDistanceFilter::init()
{
    core::componentmodel::topology::BaseMeshTopology *bmt = getContext()->getMeshTopology();

    if (bmt != 0)
    {
        helper::vector< PointInfo >& pInfo = *(m_pointInfo.beginEdit());
        pInfo.resize(bmt->getNbPoints());
        m_pointInfo.endEdit();

        helper::vector< LineInfo >& lInfo = *(m_lineInfo.beginEdit());
        lInfo.resize(bmt->getNbEdges());
        m_lineInfo.endEdit();
    }
}



void LineLocalMinDistanceFilter::handleTopologyChange()
{
    core::componentmodel::topology::BaseMeshTopology *bmt = getContext()->getMeshTopology();

    assert(bmt != 0);

    std::list< const core::componentmodel::topology::TopologyChange * >::const_iterator itBegin = bmt->firstChange();
    std::list< const core::componentmodel::topology::TopologyChange * >::const_iterator itEnd = bmt->lastChange();

    m_pointInfo.handleTopologyEvents(itBegin, itEnd);
    m_lineInfo.handleTopologyEvents(itBegin, itEnd);
}



void LineLocalMinDistanceFilter::LMDFilterPointCreationFunction(int, void *param, PointInfo &pInfo, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >&)
{
    const PointLocalMinDistanceFilter *pLMDFilter = static_cast< const PointLocalMinDistanceFilter * >(param);
    pInfo.setLMDFilters(pLMDFilter);
}



void LineLocalMinDistanceFilter::LMDFilterLineCreationFunction(int, void *param, LineInfo &lInfo, const topology::Edge&, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >&)
{
    const LineLocalMinDistanceFilter *lLMDFilter = static_cast< const LineLocalMinDistanceFilter * >(param);
    lInfo.setLMDFilters(lLMDFilter);
}



SOFA_DECL_CLASS(LineLocalMinDistanceFilter)

int LineLocalMinDistanceFilterClass = core::RegisterObject("This class manages Line collision models cones filters computations and updates.")
        .add< LineLocalMinDistanceFilter >()
        ;

} // namespace collision

} // namespace component

} // namespace sofa
