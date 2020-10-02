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
#ifndef SOFA_MANIFOLD_TOPOLOGY_EDGESETGEOMETRYALGORITHMS_H
#define SOFA_MANIFOLD_TOPOLOGY_EDGESETGEOMETRYALGORITHMS_H

#include <ManifoldTopologies/config.h>
#include <SofaBaseTopology/EdgeSetGeometryAlgorithms.h>

namespace sofa
{

namespace component
{

namespace topology
{
using core::topology::BaseMeshTopology;
typedef BaseMeshTopology::EdgeID EdgeID;
typedef BaseMeshTopology::Edge Edge;
typedef BaseMeshTopology::SeqEdges SeqEdges;
typedef BaseMeshTopology::EdgesAroundVertex EdgesAroundVertex;

/**
* A class that provides geometry information on an ManifoldEdgeSet.
*/
template < class DataTypes >
class SOFA_MANIFOLD_TOPOLOGIES_API ManifoldEdgeSetGeometryAlgorithms : public EdgeSetGeometryAlgorithms<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ManifoldEdgeSetGeometryAlgorithms,DataTypes),SOFA_TEMPLATE(EdgeSetGeometryAlgorithms,DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;

    ManifoldEdgeSetGeometryAlgorithms()
        : EdgeSetGeometryAlgorithms<DataTypes>()
    {}

    virtual ~ManifoldEdgeSetGeometryAlgorithms() {}
};

#if !defined(SOFA_MANIFOLD_TOPOLOGY_EDGESETGEOMETRYALGORITHMS_CPP)
extern template class SOFA_MANIFOLD_TOPOLOGIES_API ManifoldEdgeSetGeometryAlgorithms<sofa::defaulttype::Vec3Types>;
extern template class SOFA_MANIFOLD_TOPOLOGIES_API ManifoldEdgeSetGeometryAlgorithms<sofa::defaulttype::Vec2Types>;
extern template class SOFA_MANIFOLD_TOPOLOGIES_API ManifoldEdgeSetGeometryAlgorithms<sofa::defaulttype::Vec1Types>;
extern template class SOFA_MANIFOLD_TOPOLOGIES_API ManifoldEdgeSetGeometryAlgorithms<sofa::defaulttype::Rigid3Types>;
extern template class SOFA_MANIFOLD_TOPOLOGIES_API ManifoldEdgeSetGeometryAlgorithms<sofa::defaulttype::Rigid2Types>;
#endif

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_MANIFOLD_TOPOLOGY_EDGESETGEOMETRYALGORITHMS_H
