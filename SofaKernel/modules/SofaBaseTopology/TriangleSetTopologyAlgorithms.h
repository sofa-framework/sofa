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
#ifndef SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYALGORITHMS_H
#include "config.h"

#include <SofaBaseTopology/EdgeSetTopologyAlgorithms.h>

namespace sofa
{
namespace component
{
namespace topology
{
class TriangleSetTopologyContainer;

class TriangleSetTopologyModifier;

template < class DataTypes >
class TriangleSetGeometryAlgorithms;


/**
* A class that performs topology algorithms on an TriangleSet.
*/
template < class DataTypes >
class TriangleSetTopologyAlgorithms : public EdgeSetTopologyAlgorithms<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TriangleSetTopologyAlgorithms,DataTypes), SOFA_TEMPLATE(EdgeSetTopologyAlgorithms,DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecCoord VecDeriv;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;


    typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef core::topology::BaseMeshTopology::PointID PointID;
    typedef core::topology::BaseMeshTopology::EdgeID EdgeID;
    typedef core::topology::BaseMeshTopology::ElemID ElemID;
    typedef core::topology::BaseMeshTopology::TriangleID TriangleID;
    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef core::topology::BaseMeshTopology::TrianglesAroundVertex TrianglesAroundVertex;
    typedef core::topology::BaseMeshTopology::TrianglesAroundEdge TrianglesAroundEdge;
    typedef core::topology::BaseMeshTopology::EdgesInTriangle EdgesInTriangle;


protected:
    TriangleSetTopologyAlgorithms()
        : EdgeSetTopologyAlgorithms<DataTypes>()
        , m_listTriRemove( initData(&m_listTriRemove,  "RemoveTrianglesByIndex", "Debug : Remove a triangle or a list of triangles by using their indices (only while animate)."))
        , m_listTriAdd( initData(&m_listTriAdd,  "addTrianglesByIndex", "Debug : Add a triangle or a list of triangles by using their indices (only while animate)."))
    {
    }

    virtual ~TriangleSetTopologyAlgorithms() {}
public:
    void init() override;

    void reinit() override;

   


protected:
    Data< sofa::helper::vector< TriangleID> > m_listTriRemove; ///< Debug : Remove a triangle or a list of triangles by using their indices (only while animate).
    Data< sofa::helper::vector< Triangle> > m_listTriAdd; ///< Debug : Add a triangle or a list of triangles by using their indices (only while animate).

private:
    TriangleSetTopologyContainer*				m_container;
    TriangleSetTopologyModifier*				m_modifier;
    TriangleSetGeometryAlgorithms< DataTypes >*		m_geometryAlgorithms;
};


#if !defined(SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYALGORITHMS_CPP)
extern template class SOFA_BASE_TOPOLOGY_API TriangleSetTopologyAlgorithms<defaulttype::Vec3Types>;
extern template class SOFA_BASE_TOPOLOGY_API TriangleSetTopologyAlgorithms<defaulttype::Vec2Types>;
extern template class SOFA_BASE_TOPOLOGY_API TriangleSetTopologyAlgorithms<defaulttype::Vec1Types>;
//extern template class SOFA_BASE_TOPOLOGY_API TriangleSetTopologyAlgorithms<defaulttype::Rigid3Types>;
//extern template class SOFA_BASE_TOPOLOGY_API TriangleSetTopologyAlgorithms<defaulttype::Rigid2Types>;


#endif

} // namespace topology

} // namespace component

} // namespace sofa

#endif
