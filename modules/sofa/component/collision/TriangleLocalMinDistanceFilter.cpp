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

#include <sofa/component/collision/TriangleLocalMinDistanceFilter.h>
#include <sofa/component/collision/LocalMinDistanceFilter.inl>
#include <sofa/component/topology/EdgeData.inl>
#include <sofa/component/topology/PointData.inl>
#include <sofa/component/topology/TriangleData.inl>

#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace collision
{

void TriangleInfo::buildFilter(const Triangle &t)
{
    if (isRigid())
    {
        // update rigid
    }
    else
    {
        const Vector3& pt1 = t.p1();
        const Vector3& pt2 = t.p2();
        const Vector3& pt3 = t.p3();

        m_normal = cross(pt2-pt1, pt3-pt1);
    }

    setValid();
}



bool TriangleInfo::validate(const Triangle &t, const defaulttype::Vector3 &PQ)
{
    if (isValid())
    {
        return ( (m_normal * PQ) >= 0.0 );
    }
    else
    {
        buildFilter(t);
        return validate(t, PQ);
    }
}



void TriangleLocalMinDistanceFilter::init()
{
    core::componentmodel::topology::BaseMeshTopology *bmt = getContext()->getMeshTopology();

    if (bmt != 0)
    {
        helper::vector< PointInfo >& pInfo = *(m_pointInfo.beginEdit());
        pInfo.resize(bmt->getNbPoints());
        m_pointInfo.endEdit();

        m_pointInfo.setCreateFunction(LMDFilterPointCreationFunction);
        m_pointInfo.setCreateParameter((void *) this);

        helper::vector< LineInfo >& lInfo = *(m_lineInfo.beginEdit());
        lInfo.resize(bmt->getNbEdges());
        m_lineInfo.endEdit();

        m_lineInfo.setCreateFunction(LMDFilterLineCreationFunction);
        m_lineInfo.setCreateParameter((void *) this);

        helper::vector< TriangleInfo >& tInfo = *(m_triangleInfo.beginEdit());
        tInfo.resize(bmt->getNbTriangles());
        m_triangleInfo.endEdit();

        m_triangleInfo.setCreateFunction(LMDFilterTriangleCreationFunction);
        m_triangleInfo.setCreateParameter((void *) this);
    }
}



void TriangleLocalMinDistanceFilter::handleTopologyChange()
{
    core::componentmodel::topology::BaseMeshTopology *bmt = getContext()->getMeshTopology();

    assert(bmt != 0);

    std::list< const core::componentmodel::topology::TopologyChange * >::const_iterator itBegin = bmt->firstChange();
    std::list< const core::componentmodel::topology::TopologyChange * >::const_iterator itEnd = bmt->lastChange();

    m_pointInfo.handleTopologyEvents(itBegin, itEnd);
    m_lineInfo.handleTopologyEvents(itBegin, itEnd);
    m_triangleInfo.handleTopologyEvents(itBegin, itEnd);
}



void TriangleLocalMinDistanceFilter::LMDFilterPointCreationFunction(int, void *param, PointInfo &pInfo, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >&)
{
    const PointLocalMinDistanceFilter *pLMDFilter = static_cast< const PointLocalMinDistanceFilter * >(param);
    pInfo.setLMDFilters(pLMDFilter);
}



void TriangleLocalMinDistanceFilter::LMDFilterLineCreationFunction(int, void *param, LineInfo &lInfo, const topology::Edge&, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >&)
{
    const LineLocalMinDistanceFilter *lLMDFilter = static_cast< const LineLocalMinDistanceFilter * >(param);
    lInfo.setLMDFilters(lLMDFilter);
}



void TriangleLocalMinDistanceFilter::LMDFilterTriangleCreationFunction(int, void *param, TriangleInfo &tInfo, const topology::Triangle&, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >&)
{
    const TriangleLocalMinDistanceFilter *tLMDFilter = static_cast< const TriangleLocalMinDistanceFilter * >(param);
    tInfo.setLMDFilters(tLMDFilter);
}



SOFA_DECL_CLASS(TriangleLocalMinDistanceFilter)

int TriangleLocalMinDistanceFilterClass = core::RegisterObject("This class manages Triangle collision models cones filters computations and updates.")
        .add< TriangleLocalMinDistanceFilter >()
        ;

} // namespace collision

} // namespace component

} // namespace sofa
