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
#ifndef SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGY_H

#include <sofa/component/topology/EdgeSetTopology.h>

#include <sofa/component/topology/TriangleSetGeometryAlgorithms.h>
#include <sofa/component/topology/TriangleSetTopologyAlgorithms.h>
#include <sofa/component/topology/TriangleSetTopologyModifier.h>
#include <sofa/component/topology/TriangleSetTopologyContainer.h>
#include <sofa/component/topology/TriangleSetTopologyChange.h>

namespace sofa
{
namespace component
{
namespace topology
{
// forward declarations
template<class DataTypes>
class TriangleSetTopology;

class TriangleSetTopologyContainer;

template<class DataTypes>
class TriangleSetTopologyModifier;

template < class DataTypes >
class TriangleSetTopologyAlgorithms;

template < class DataTypes >
class TriangleSetGeometryAlgorithms;

template <class DataTypes>
class TriangleSetTopologyLoader;

class TrianglesAdded;
class TrianglesRemoved;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::TriangleID TriangleID;
typedef BaseMeshTopology::Triangle Triangle;
typedef BaseMeshTopology::SeqTriangles SeqTriangles;
typedef BaseMeshTopology::VertexTriangles VertexTriangles;
typedef BaseMeshTopology::EdgeTriangles EdgeTriangles;
typedef BaseMeshTopology::TriangleEdges TriangleEdges;

/** Describes a topological object that consists as a set of points and triangles connected these points */
template<class DataTypes>
class TriangleSetTopology : public EdgeSetTopology <DataTypes>
{
public:
    TriangleSetTopology(component::MechanicalObject<DataTypes> *obj);

    virtual ~TriangleSetTopology() {}

    virtual void init();

    /** \brief Returns the TriangleSetTopologyContainer object of this TriangleSetTopology.
    */
    TriangleSetTopologyContainer *getTriangleSetTopologyContainer() const
    {
        return static_cast<TriangleSetTopologyContainer *> (this->m_topologyContainer);
    }

    /** \brief Returns the TriangleSetTopologyModifier object of this TriangleSetTopology.
    */
    TriangleSetTopologyModifier<DataTypes> *getTriangleSetTopologyModifier() const
    {
        return static_cast<TriangleSetTopologyModifier<DataTypes> *> (this->m_topologyModifier);
    }

    /** \brief Returns the TriangleSetTopologyAlgorithms object of this TriangleSetTopology.
    */
    TriangleSetTopologyAlgorithms<DataTypes> *getTriangleSetTopologyAlgorithms() const
    {
        return static_cast<TriangleSetTopologyAlgorithms<DataTypes> *> (this->m_topologyAlgorithms);
    }

    /** \brief Returns the TriangleSetTopologyAlgorithms object of this TriangleSetTopology.
    */
    TriangleSetGeometryAlgorithms<DataTypes> *getTriangleSetGeometryAlgorithms() const
    {
        return static_cast<TriangleSetGeometryAlgorithms<DataTypes> *> (this->m_geometryAlgorithms);
    }

    /// BaseMeshTopology API
    /// @{

    const SeqTriangles& getTriangles()
    {
        return getTriangleSetTopologyContainer()->getTriangleArray();
    }

    /// Returns the set of edges adjacent to a given triangle.
    const TriangleEdges& getEdgeTriangleShell(TriangleID i)
    {
        return getTriangleSetTopologyContainer()->getTriangleEdge(i);
    }

    /// Returns the set of triangles adjacent to a given vertex.
    const VertexTriangles& getTriangleVertexShell(PointID i)
    {
        return getTriangleSetTopologyContainer()->getTriangleVertexShell(i);
    }

    /// Returns the set of triangles adjacent to a given edge.
    const EdgeTriangles& getTriangleEdgeShell(EdgeID i)
    {
        return getTriangleSetTopologyContainer()->getTriangleEdgeShell(i);
    }

    /// Returns the index of the triangle given three vertex indices; returns -1 if no edge exists
    int getTriangleIndex(PointID v1, PointID v2, PointID v3)
    {
        return  getTriangleSetTopologyContainer()->getTriangleIndex(v1, v2, v3);
    }

    /// Returns the index (either 0, 1 ,2) of the vertex whose global index is vertexIndex. Returns -1 if none
    int getVertexIndexInTriangle(const Triangle &t, PointID i) const
    {
        return  getTriangleSetTopologyContainer()->getVertexIndexInTriangle(t, i);
    }
    /// Returns the index (either 0, 1 ,2) of the edge whose global index is edgeIndex. Returns -1 if none
    int getEdgeIndexInTriangle(const TriangleEdges &t, EdgeID i) const
    {
        return  getTriangleSetTopologyContainer()->getEdgeIndexInTriangle(t, i);
    }

    /// @}

protected:
    virtual void createComponents();
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
