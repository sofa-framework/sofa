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
#ifndef SOFA_COMPONENT_TOPOLOGY_QUADSETTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_QUADSETTOPOLOGY_H

#include <sofa/component/topology/EdgeSetTopology.h>

#include <sofa/component/topology/QuadSetGeometryAlgorithms.h>
#include <sofa/component/topology/QuadSetTopologyAlgorithms.h>
#include <sofa/component/topology/QuadSetTopologyModifier.h>
#include <sofa/component/topology/QuadSetTopologyContainer.h>
#include <sofa/component/topology/QuadSetTopologyChange.h>

namespace sofa
{
namespace component
{
namespace topology
{
// forward declarations
template <class DataTypes>
class QuadSetTopology;

class QuadSetTopologyContainer;

template <class DataTypes>
class QuadSetTopologyModifier;

template < class DataTypes >
class QuadSetTopologyAlgorithms;

template < class DataTypes >
class QuadSetGeometryAlgorithms;

template <class DataTypes>
class QuadSetTopologyLoader;

class QuadsAdded;
class QuadsRemoved;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::QuadID QuadID;
typedef BaseMeshTopology::Quad Quad;
typedef BaseMeshTopology::SeqQuads SeqQuads;
typedef BaseMeshTopology::VertexQuads VertexQuads;
typedef BaseMeshTopology::EdgeQuads EdgeQuads;
typedef BaseMeshTopology::QuadEdges QuadEdges;

/** Describes a topological object that consists as a set of points and quads connected these points */
template<class DataTypes>
class QuadSetTopology : public EdgeSetTopology <DataTypes>
{
public:

    QuadSetTopology(component::MechanicalObject<DataTypes> *obj);

    virtual ~QuadSetTopology() {}

    virtual void init();

    /** \brief Returns the QuadSetTopologyContainer object of this QuadSetTopology.
    */
    QuadSetTopologyContainer *getQuadSetTopologyContainer() const
    {
        return static_cast<QuadSetTopologyContainer *> (this->m_topologyContainer);
    }

    /** \brief Returns the QuadSetTopologyModifier object of this QuadSetTopology.
    */
    QuadSetTopologyModifier<DataTypes> *getQuadSetTopologyModifier() const
    {
        return static_cast<QuadSetTopologyModifier<DataTypes> *> (this->m_topologyModifier);
    }

    /** \brief Returns the QuadSetTopologyAlgorithms object of this QuadSetTopology.
    */
    QuadSetTopologyAlgorithms<DataTypes> *getQuadSetTopologyAlgorithms() const
    {
        return static_cast<QuadSetTopologyAlgorithms<DataTypes> *> (this->m_topologyAlgorithms);
    }

    /** \brief Returns the QuadSetTopologyAlgorithms object of this QuadSetTopology.
    */
    QuadSetGeometryAlgorithms<DataTypes> *getQuadSetGeometryAlgorithms() const
    {
        return static_cast<QuadSetGeometryAlgorithms<DataTypes> *> (this->m_geometryAlgorithms);
    }

    /// BaseMeshTopology API
    /// @{

    const SeqQuads& getQuads()   { return getQuadSetTopologyContainer()->getQuadArray(); }
    /// Returns the set of edges adjacent to a given quad.
    const QuadEdges& getEdgeQuadShell(QuadID i) { return getQuadSetTopologyContainer()->getQuadEdge(i); }
    /// Returns the set of quads adjacent to a given vertex.
    const VertexQuads& getQuadVertexShell(PointID i) { return getQuadSetTopologyContainer()->getQuadVertexShell(i); }
    /// Returns the set of quads adjacent to a given edge.
    const EdgeQuads& getQuadEdgeShell(EdgeID i) { return getQuadSetTopologyContainer()->getQuadEdgeShell(i); }

    /// Returns the index of the quad given four vertex indices; returns -1 if no edge exists
    int getQuadIndex(PointID v1, PointID v2, PointID v3, PointID v4)
    {
        return getQuadSetTopologyContainer()->getQuadIndex(v1, v2, v3, v4);
    }

    /// Returns the index (either 0, 1 ,2 or 3) of the vertex whose global index is vertexIndex. Returns -1 if none
    int getVertexIndexInQuad(Quad &t, PointID i) const
    {
        return getQuadSetTopologyContainer()->getVertexIndexInQuad(t, i);
    }
    /// Returns the index (either 0, 1 ,2, 3) of the edge whose global index is edgeIndex. Returns -1 if none
    int getEdgeIndexInQuad(QuadEdges &t, EdgeID i) const
    {
        return getQuadSetTopologyContainer()->getEdgeIndexInQuad(t, i);
    }

    /// @}

protected:
    virtual void createComponents();
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
