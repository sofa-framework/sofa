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
#ifndef SOFA_COMPONENT_TOPOLOGY_MANIFOLDTRIANGLESETTOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_MANIFOLDTRIANGLESETTOPOLOGYCONTAINER_H

#include <sofa/component/topology/TriangleSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace topology
{
/**To Do
   modify TriangleSetTopologyModifier in ManifoldTriangleSetTopologyModifier
*/
class TriangleSetTopologyModifier;

using core::componentmodel::topology::BaseMeshTopology;

typedef BaseMeshTopology::PointID			PointID;
typedef BaseMeshTopology::EdgeID			EdgeID;
typedef BaseMeshTopology::TriangleID		TriangleID;
typedef BaseMeshTopology::Edge				Edge;
typedef BaseMeshTopology::Triangle			Triangle;
typedef BaseMeshTopology::SeqTriangles		SeqTriangles;
typedef BaseMeshTopology::TriangleEdges		TriangleEdges;
typedef BaseMeshTopology::VertexTriangles	VertexTriangles;
typedef BaseMeshTopology::EdgeTriangles		EdgeTriangles;


/** A class that stores a set of triangles and provides access
to each triangle, triangle's edges and vertices.
This topology is contraint by the manifold property: each edge is adjacent to either one or at most two Triangles.*/
class SOFA_COMPONENT_CONTAINER_API ManifoldTriangleSetTopologyContainer : public TriangleSetTopologyContainer
{
    /**To Do
       modify TriangleSetTopologyModifier in ManifoldTriangleSetTopologyModifier
    */
    friend class TriangleSetTopologyModifier;

public:

    ManifoldTriangleSetTopologyContainer();

    ManifoldTriangleSetTopologyContainer(const sofa::helper::vector< Triangle > &triangles );

    virtual ~ManifoldTriangleSetTopologyContainer() {}

    virtual void init();

    virtual void clear();

    /** \brief Checks if the topology is coherent
    *
    * Check if the shell arrays are coherent
    * If all the triangles from arrays are detected in the shell arrays
    * Check if the Topology is manifold.
    */
    virtual bool checkTopology() const;


    /**To Do
       Add exceptions throw in all functions
    */

    /** \brief: Given a Triangle and a Vertex i, returns the next adjacent triangle to the first one
    * in the counterclockwise direction around the ith vertex.
    *
    * return -1 if there is no adjacent triangle in this direction.
    * return -2 if the vertex does not belongs to this Triangle or if there is an other error.
    */
    int getNextTriangleVertexShell(PointID vertexIndex, TriangleID triangleIndex);


    /** \brief: Given a Triangle and a Vertex i, returns the next adjacent triangle to this first one
    * in the clockwise direction around the ith vertex.
    *
    * return -1 if there is no adjacent triangle in this direction
    * return -2 if the vertex does not belongs to this Triangle or if there is an other error.
    */
    int getPreviousTriangleVertexShell(PointID vertexIndex, TriangleID triangleIndex);


    /** \brief: Given a Triangle and a Edge i, returns the other adjacent triangle to the ith edge.
    *
    * return -1 if there is only one triangle adjacent to this edge.
    * return -2 if the edge does not belongs to this Triangle or if there is an other error.
    */
    int getOppositeTriangleEdgeShell(EdgeID edgeIndex, TriangleID triangleIndex);


    /** \brief: Given a Edge and a Vertex i, returns the next edge containing the ith vertex
    * in the counterclockwise direction around the ith vertex.
    *
    * return -1 if there is adjacent no triangle in this direction
    * return -2 if the vertex does not belongs to the edge or if there is an other error.
    */
    int getNextEdgeVertexShell(PointID vertexIndex, EdgeID edgeIndex);


    /** \brief: Given a Edge and a Vertex i, returns the next edge containing the ith vertex
    * in the clockwise direction around the ith vertex.
    *
    * return -1 if there is no triangle in this direction
    * return -2 if the vertex does not belongs to the edge or if there is an other error.
    */
    int getPreviousEdgeVertexShell(PointID vertexIndex, EdgeID edgeIndex);

    /** \brief: Return a vector of TriangleID which are on a border. I.e which have at least
    * one edge not adjacent to an other Triangle.
    * To Do: For the moment use TriangleEdgeShellArray(), check if has to be reimplemented in an other way
    */
    sofa::helper::vector <TriangleID> getTrianglesBorder();

    /** \brief: Return a vector of EdgeID which are on a border. I.e which are adjacent to only one Triangle.
    * To Do: For the moment use TriangleEdgeShellArray(), check if has to be reimplemented in an other way
    */
    sofa::helper::vector <EdgeID> getEdgesBorder();

protected:

    /** \brief Creates the EdgeSet array.
    *
    * Create the set of edges.
    * Function derived from TriangleSetTopologyContainer::createEdgeSetArray().
    * In this function, vertices of each edge are not stored in lexicographic order.
    *
    */
    virtual void createEdgeSetArray();


    /** \brief Creates the Edge Vertex Shell Array.
    *
    * This function is only called if the EdgeVertexShell array is required.
    * m_EdgeVertexShell[i] contains the indices of all edges adjacent to the ith vertex.
    * This funciton check if there are T connections between more than 2 edges at the ith DOF.
    *
    */
    virtual void createEdgeVertexShellArray();


    /** \brief Creates the Triangle Vertex Shell Array
    *
    * This function is only called if the TriangleVertexShell array is required.
    * m_triangleVertexShell[i] contains the indices of all triangles adjacent to the ith vertex.
    * This function check if there are T connections between more than 3 triangles at the ith DOF.
    *
    */
    virtual void createTriangleVertexShellArray();


    /** \brief Creates the Triangle Edge Shell Array
    *
    * This function is only called if the TriangleEdgeShell array is required.
    * m_triangleEdgeShell[i] contains the indices of all triangles adjacent to the ith edge.
    * This function check if there are more than 2 triangles adjacent to each edge.
    *
    */
    virtual void createTriangleEdgeShellArray();

};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
