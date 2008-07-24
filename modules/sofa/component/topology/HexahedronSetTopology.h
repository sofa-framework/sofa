/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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

// CONVENTION : indices ordering for the vertices of an hexahedron :
//
// 	   Y  3---------2
//     ^ /	       /|
//     |/	      / |
//     7---------6  |
//     |    	 |  |
//     |  0------|--1
//     | / 	     | /
//     |/	     |/
//     4---------5-->X
//    /
//   /
//  Z

#ifndef SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGY_H

#include <sofa/component/topology/QuadSetTopology.h>

#include <sofa/component/topology/HexahedronSetGeometryAlgorithms.h>
#include <sofa/component/topology/HexahedronSetTopologyAlgorithms.h>
#include <sofa/component/topology/HexahedronSetTopologyModifier.h>
#include <sofa/component/topology/HexahedronSetTopologyContainer.h>
#include <sofa/component/topology/HexahedronSetTopologyChange.h>

namespace sofa
{
namespace component
{
namespace topology
{
// forward declarations
template <class DataTypes>
class HexahedronSetTopology;

class HexahedronSetTopologyContainer;

template <class DataTypes>
class HexahedronSetTopologyModifier;

template < class DataTypes >
class HexahedronSetTopologyAlgorithms;

template < class DataTypes >
class HexahedronSetGeometryAlgorithms;

template <class DataTypes>
class HexahedronSetTopologyLoader;

class HexahedraAdded;
class HexahedraRemoved;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::HexaID HexaID;
typedef BaseMeshTopology::Hexa Hexa;
typedef BaseMeshTopology::SeqHexas SeqHexas;
typedef BaseMeshTopology::VertexHexas VertexHexas;
typedef BaseMeshTopology::EdgeHexas EdgeHexas;
typedef BaseMeshTopology::QuadHexas QuadHexas;
typedef BaseMeshTopology::HexaEdges HexaEdges;
typedef BaseMeshTopology::HexaQuads HexaQuads;

typedef Hexa Hexahedron;
typedef HexaEdges HexahedronEdges;
typedef HexaQuads HexahedronQuads;

/** Describes a topological object that consists as a set of points and hexahedra connected these points */
template<class DataTypes>
class HexahedronSetTopology : public QuadSetTopology <DataTypes>
{
public:
    HexahedronSetTopology(component::MechanicalObject<DataTypes> *obj);

    virtual ~HexahedronSetTopology() {}

    virtual void init();

    /** \brief Returns the HexahedronSetTopologyContainer object of this HexahedronSetTopology.
    */
    HexahedronSetTopologyContainer *getHexahedronSetTopologyContainer() const
    {
        return static_cast<HexahedronSetTopologyContainer *> (this->m_topologyContainer);
    }

    /** \brief Returns the HexahedronSetTopologyModifier object of this HexahedronSetTopology.
    */
    HexahedronSetTopologyModifier<DataTypes> *getHexahedronSetTopologyModifier() const
    {
        return static_cast<HexahedronSetTopologyModifier<DataTypes> *> (this->m_topologyModifier);
    }

    /** \brief Returns the HexahedronSetTopologyAlgorithms object of this HexahedronSetTopology.
    */
    HexahedronSetTopologyAlgorithms<DataTypes> *getHexahedronSetTopologyAlgorithms() const
    {
        return static_cast<HexahedronSetTopologyAlgorithms<DataTypes> *> (this->m_topologyAlgorithms);
    }

    /** \brief Returns the HexahedronSetTopologyAlgorithms object of this HexahedronSetTopology.
    */
    HexahedronSetGeometryAlgorithms<DataTypes> *getHexahedronSetGeometryAlgorithms() const
    {
        return static_cast<HexahedronSetGeometryAlgorithms<DataTypes> *> (this->m_geometryAlgorithms);
    }

    /// BaseMeshTopology API
    /// @{

    const SeqHexas& getHexas()   { return getHexahedronSetTopologyContainer()->getHexahedronArray(); }
    /// Returns the set of edges adjacent to a given hexahedron.
    const HexaEdges& getEdgeHexaShell(HexaID i) { return getHexahedronSetTopologyContainer()->getHexahedronEdges(i); }
    /// Returns the set of quads adjacent to a given hexahedron.
    const HexaQuads& getQuadHexaShell(HexaID i) { return getHexahedronSetTopologyContainer()->getHexahedronQuads(i); }
    /// Returns the set of hexahedra adjacent to a given vertex.
    const VertexHexas& getHexaVertexShell(PointID i) { return getHexahedronSetTopologyContainer()->getHexahedronVertexShell(i); }
    /// Returns the set of hexahedra adjacent to a given edge.
    const EdgeHexas& getHexaEdgeShell(EdgeID i) { return getHexahedronSetTopologyContainer()->getHexahedronEdgeShell(i); }
    /// Returns the set of hexahedra adjacent to a given quad.
    const QuadHexas& getHexaQuadShell(QuadID i) { return getHexahedronSetTopologyContainer()->getHexahedronQuadShell(i); }

    /// Returns the index of the hexahedron given eight vertex indices; returns -1 if no edge exists
    int getHexahedronIndex(PointID v1, PointID v2, PointID v3, PointID v4, PointID v5, PointID v6, PointID v7, PointID v8)
    {
        return getHexahedronSetTopologyContainer()->getHexahedronIndex(v1, v2, v3, v4, v5, v6, v7, v8);
    }

    /// Returns the index (either 0, 1 ,2, 3, 4, 5, 6, or 7) of the vertex whose global index is vertexIndex. Returns -1 if none
    int getVertexIndexInHexahedron(Hexa &t, PointID i) const
    {
        return getHexahedronSetTopologyContainer()->getVertexIndexInHexahedron(t, i);
    }
    /// Returns the index (either 0, 1 ,2 ,3, 4, 5, 6, 7, 8, 9, 10, 11) of the edge whose global index is edgeIndex. Returns -1 if none
    int getEdgeIndexInHexahedron(const HexaEdges &t, EdgeID i) const
    {
        return getHexahedronSetTopologyContainer()->getEdgeIndexInHexahedron(t, i);
    }
    /// Returns the index (either 0, 1 ,2 ,3, 4, 5) of the quad whose global index is quadIndex. Returns -1 if none
    int getQuadIndexInHexahedron(const HexaQuads &t, QuadID i) const
    {
        return getHexahedronSetTopologyContainer()->getQuadIndexInHexahedron(t, i);
    }


    /// @}

protected:
    virtual void createComponents();
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
