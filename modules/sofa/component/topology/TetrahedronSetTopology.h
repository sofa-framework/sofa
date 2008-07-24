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
#ifndef SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGY_H

#include <sofa/component/topology/TriangleSetTopology.h>

#include <sofa/component/topology/TetrahedronSetGeometryAlgorithms.h>
#include <sofa/component/topology/TetrahedronSetTopologyAlgorithms.h>
#include <sofa/component/topology/TetrahedronSetTopologyModifier.h>
#include <sofa/component/topology/TetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/TetrahedronSetTopologyChange.h>

namespace sofa
{

namespace component
{

namespace topology
{
// forward declarations
template <class DataTypes>
class TetrahedronSetTopology;

class TetrahedronSetTopologyContainer;

template <class DataTypes>
class TetrahedronSetTopologyModifier;

template < class DataTypes >
class TetrahedronSetTopologyAlgorithms;

template < class DataTypes >
class TetrahedronSetGeometryAlgorithms;

template <class DataTypes>
class TetrahedronSetTopologyLoader;

class TetrahedraAdded;
class TetrahedraRemoved;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::TetraID TetraID;
typedef BaseMeshTopology::Tetra Tetra;
typedef BaseMeshTopology::SeqTetras SeqTetras;
typedef BaseMeshTopology::VertexTetras VertexTetras;
typedef BaseMeshTopology::EdgeTetras EdgeTetras;
typedef BaseMeshTopology::TriangleTetras TriangleTetras;
typedef BaseMeshTopology::TetraEdges TetraEdges;
typedef BaseMeshTopology::TetraTriangles TetraTriangles;

typedef Tetra Tetrahedron;
typedef TetraEdges TetrahedronEdges;
typedef TetraTriangles TetrahedronTriangles;

/** Describes a topological object that consists as a set of points and tetrahedra connected these points */
template<class DataTypes>
class TetrahedronSetTopology : public TriangleSetTopology <DataTypes>
{
public:
    TetrahedronSetTopology(component::MechanicalObject<DataTypes> *obj);

    virtual ~TetrahedronSetTopology() {}

    virtual void init();

    /** \brief Returns the TetrahedronSetTopologyContainer object of this TetrahedronSetTopology.
    */
    TetrahedronSetTopologyContainer *getTetrahedronSetTopologyContainer() const
    {
        return static_cast<TetrahedronSetTopologyContainer *> (this->m_topologyContainer);
    }

    /** \brief Returns the TetrahedronSetTopologyModifier object of this TetrahedronSetTopology.
    */
    TetrahedronSetTopologyModifier<DataTypes> *getTetrahedronSetTopologyModifier() const
    {
        return static_cast<TetrahedronSetTopologyModifier<DataTypes> *> (this->m_topologyModifier);
    }

    /** \brief Returns the TetrahedronSetTopologyAlgorithms object of this TetrahedronSetTopology.
    */
    TetrahedronSetTopologyAlgorithms<DataTypes> *getTetrahedronSetTopologyAlgorithms() const
    {
        return static_cast<TetrahedronSetTopologyAlgorithms<DataTypes> *> (this->m_topologyAlgorithms);
    }

    /** \brief Returns the TetrahedronSetTopologyAlgorithms object of this TetrahedronSetTopology.
    */
    TetrahedronSetGeometryAlgorithms<DataTypes> *getTetrahedronSetGeometryAlgorithms() const
    {
        return static_cast<TetrahedronSetGeometryAlgorithms<DataTypes> *> (this->m_geometryAlgorithms);
    }

    /// BaseMeshTopology API
    /// @{

    const SeqTetras& getTetras()   { return getTetrahedronSetTopologyContainer()->getTetrahedronArray(); }
    /// Returns the set of edges adjacent to a given tetrahedron.
    const TetraEdges& getEdgeTetraShell(TetraID i) { return getTetrahedronSetTopologyContainer()->getTetrahedronEdges(i); }
    /// Returns the set of triangles adjacent to a given tetrahedron.
    const TetraTriangles& getTriangleTetraShell(TetraID i) { return getTetrahedronSetTopologyContainer()->getTetrahedronTriangles(i); }
    /// Returns the set of tetrahedra adjacent to a given vertex.
    const VertexTetras& getTetraVertexShell(PointID i) { return getTetrahedronSetTopologyContainer()->getTetrahedronVertexShell(i); }
    /// Returns the set of tetrahedra adjacent to a given edge.
    const EdgeTetras& getTetraEdgeShell(EdgeID i) { return getTetrahedronSetTopologyContainer()->getTetrahedronEdgeShell(i); }
    /// Returns the set of tetrahedra adjacent to a given triangle.
    const TriangleTetras& getTetraTriangleShell(TriangleID i) { return getTetrahedronSetTopologyContainer()->getTetrahedronTriangleShell(i); }

    /// Returns the index of the tetrahedron given four vertex indices; returns -1 if no edge exists
    int getTetrahedronIndex(PointID v1, PointID v2, PointID v3, PointID v4)
    {
        return getTetrahedronSetTopologyContainer()->getTetrahedronIndex(v1, v2, v3, v4);
    }

    /// Returns the index (either 0, 1 ,2 or 3) of the vertex whose global index is vertexIndex. Returns -1 if none
    int getVertexIndexInTetrahedron(const Tetra &t, PointID i) const
    {
        return getTetrahedronSetTopologyContainer()->getVertexIndexInTetrahedron(t, i);
    }
    /// Returns the index (either 0, 1 ,2 ,3, 4, 5) of the edge whose global index is edgeIndex. Returns -1 if none
    int getEdgeIndexInTetrahedron(const TetraEdges &t, EdgeID i) const
    {
        return getTetrahedronSetTopologyContainer()->getEdgeIndexInTetrahedron(t, i);
    }
    /// Returns the index (either 0, 1 ,2 ,3) of the triangle whose global index is triangleIndex. Returns -1 if none
    int getTriangleIndexInTetrahedron(const TetraTriangles &t, TriangleID i) const
    {
        return getTetrahedronSetTopologyContainer()->getTriangleIndexInTetrahedron(t, i);
    }
    /// Returns for each index (between 0 and 5) the two vertex indices that are adjacent to that edge
    Edge getLocalTetrahedronEdges (const unsigned int i) const
    {
        return getTetrahedronSetTopologyContainer()->getLocalTetrahedronEdges(i);
    }

    /// @}

protected:
    virtual void createComponents();
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
