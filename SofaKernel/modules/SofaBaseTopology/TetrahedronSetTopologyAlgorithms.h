/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGYALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGYALGORITHMS_H
#include "config.h"

#include <SofaBaseTopology/TriangleSetTopologyAlgorithms.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace topology
{
class TetrahedronSetTopologyContainer;

class TetrahedronSetTopologyModifier;

template < class DataTypes >
class TetrahedronSetGeometryAlgorithms;

/**
* A class that performs topology algorithms on an TetrahedronSet.
*/
template < class DataTypes >
class TetrahedronSetTopologyAlgorithms : public TriangleSetTopologyAlgorithms<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TetrahedronSetTopologyAlgorithms,DataTypes),SOFA_TEMPLATE(TriangleSetTopologyAlgorithms,DataTypes));

    typedef typename DataTypes::Real Real;
    typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef core::topology::BaseMeshTopology::EdgeID EdgeID;
    typedef core::topology::BaseMeshTopology::TetraID TetraID;
    typedef core::topology::BaseMeshTopology::Tetra Tetra;
    typedef core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;
    typedef core::topology::BaseMeshTopology::EdgesInTetrahedron EdgesInTetrahedron;
    typedef core::topology::BaseMeshTopology::TetrahedraAroundEdge TetrahedraAroundEdge;
    typedef typename DataTypes::Coord Coord;
protected:
    TetrahedronSetTopologyAlgorithms()
        : TriangleSetTopologyAlgorithms<DataTypes>()
    {}

    virtual ~TetrahedronSetTopologyAlgorithms() {}
public:
    virtual void init() override;

    void removeTetra(sofa::helper::vector<TetraID>& ind_ta);

    void subDivideTetrahedronsWithPlane(sofa::helper::vector< sofa::helper::vector<double> >& coefs, sofa::helper::vector<EdgeID>& intersectedEdgeID, Coord /*planePos*/, Coord planeNormal);
    void subDivideTetrahedronsWithPlane(sofa::helper::vector<Coord>& intersectedPoints, sofa::helper::vector<EdgeID>& intersectedEdgeID, Coord planePos, Coord planeNormal);
    int subDivideTetrahedronWithPlane(TetraID tetraIdx, sofa::helper::vector<EdgeID>& intersectedEdgeID, sofa::helper::vector<unsigned int>& intersectedPointID, Coord planeNormal, sofa::helper::vector<Tetra>& toBeAddedTetra);

    void subDivideRestTetrahedronsWithPlane(sofa::helper::vector< sofa::helper::vector<double> >& coefs, sofa::helper::vector<EdgeID>& intersectedEdgeID, Coord /*planePos*/, Coord planeNormal);
    void subDivideRestTetrahedronsWithPlane(sofa::helper::vector<Coord>& intersectedPoints, sofa::helper::vector<EdgeID>& intersectedEdgeID, Coord planePos, Coord planeNormal);
    int subDivideRestTetrahedronWithPlane(TetraID tetraIdx, sofa::helper::vector<EdgeID>& intersectedEdgeID, sofa::helper::vector<unsigned int>& intersectedPointID, Coord planeNormal, sofa::helper::vector<Tetra>& toBeAddedTetra);


protected:
    TetrahedronSetTopologyContainer*					m_container;
    TetrahedronSetTopologyModifier*						m_modifier;
    TetrahedronSetGeometryAlgorithms< DataTypes >*		m_geometryAlgorithms;
    unsigned int	m_intialNbPoints;
    Real m_baryLimit;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGYALGORITHMS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetTopologyAlgorithms<defaulttype::Vec3dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetTopologyAlgorithms<defaulttype::Vec2dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetTopologyAlgorithms<defaulttype::Vec1dTypes>;
//extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetTopologyAlgorithms<defaulttype::Rigid3dTypes>;
//extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetTopologyAlgorithms<defaulttype::Rigid2dTypes>;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetTopologyAlgorithms<defaulttype::Vec3fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetTopologyAlgorithms<defaulttype::Vec2fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetTopologyAlgorithms<defaulttype::Vec1fTypes>;
//extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetTopologyAlgorithms<defaulttype::Rigid3fTypes>;
//extern template class SOFA_BASE_TOPOLOGY_API TetrahedronSetTopologyAlgorithms<defaulttype::Rigid2fTypes>;
#endif
#endif

} // namespace topology

} // namespace component

} // namespace sofa

#endif
