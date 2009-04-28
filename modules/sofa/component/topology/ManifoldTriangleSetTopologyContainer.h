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
#ifndef SOFA_COMPONENT_TOPOLOGY_MANIFOLDTRIANGLESETTOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_MANIFOLDTRIANGLESETTOPOLOGYCONTAINER_H

#include <sofa/component/topology/TriangleSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace topology
{
class ManifoldTriangleSetTopologyModifier;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::PointID		PointID;
typedef BaseMeshTopology::EdgeID		EdgeID;
typedef BaseMeshTopology::TriangleID	TriangleID;
typedef BaseMeshTopology::Edge		Edge;
typedef BaseMeshTopology::Triangle	Triangle;
typedef BaseMeshTopology::SeqTriangles	SeqTriangles;
typedef BaseMeshTopology::TriangleEdges	TriangleEdges;
typedef BaseMeshTopology::VertexTriangles	VertexTriangles;
typedef BaseMeshTopology::EdgeTriangles	EdgeTriangles;

/** A class that stores a set of triangles and provides access
to each triangle, triangle's edges and vertices.
This topology is contraint by the manifold property: each edge is adjacent to either one or at most two Triangles.*/
class SOFA_COMPONENT_CONTAINER_API ManifoldTriangleSetTopologyContainer : public TriangleSetTopologyContainer
{

    friend class ManifoldTriangleSetTopologyModifier;

public:

    ManifoldTriangleSetTopologyContainer();

    ManifoldTriangleSetTopologyContainer(const sofa::helper::vector< Triangle > &triangles );

    virtual ~ManifoldTriangleSetTopologyContainer() {}

    virtual void init();

    virtual void clear();

    /** \brief Checks if the topology is coherent.
     *
     * The function TriangleSetTopologyContainer::CheckTopology() Check if the shell arrays are coherent
     * from on to the other.
     * In this class, we check the topology. I.e for m_triangleVertexShell
     *   - Test if triangles are stocked in counterclockewise direction around each vertex and that they are contiguous.
     *   - Test if no triangles are missing and if no triangle are badly connected to a vertex.
     * For m_triangleEdgeShell
     *   - Test if no triangles are missing.
     *   - Test if there is at least 1 and not more than 2 triangles adjacent to each edge.
     *   - Test if triangles are well order in the shell: In the first one, vertices of the
     correspondant edge are in oriented in counterclockwise direction in this triangle.
     And in the clockwise direction in the second triangle (if this one exist).
     *
     */
    virtual bool checkTopology() const;


    /** \brief: Given a Triangle and a Vertex i, returns the next adjacent triangle to the first one
     * in the counterclockwise direction around the ith vertex.
     *
     * @param unsigned int, unsigned int
     * @return -1 if there is no adjacent triangle in this direction.
     * @return -2 if the vertex does not belongs to this Triangle or if there is an other error.
     */
    int getNextTriangleVertexShell(PointID vertexIndex, TriangleID triangleIndex);


    /** \brief: Given a Triangle and a Vertex i, returns the next adjacent triangle to this first one
     * in the clockwise direction around the ith vertex.
     *
     * @return -1 if there is no adjacent triangle in this direction
     * @return -2 if the vertex does not belongs to this Triangle or if there is an other error.
     */
    int getPreviousTriangleVertexShell(PointID vertexIndex, TriangleID triangleIndex);


    /** \brief: Given a Triangle and a Edge i, returns the other adjacent triangle to the ith edge.
     *
     * @return -1 if there is only one triangle adjacent to this edge.
     * @return -2 if the edge does not belongs to this Triangle or if there is an other error.
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


    /** \brief: Return a vector of TriangleID which are on a border.
     * @see createElementsOnBorder()
     */
    const sofa::helper::vector <TriangleID>& getTrianglesOnBorder();


    /** \brief: Return a vector of EdgeID which are on a border.
     * @see createElementsOnBorder()
     */
    const sofa::helper::vector <EdgeID>& getEdgesOnBorder();


    /** \brief: Return a vector of PointID which are on a border.
     * @see createElementsOnBorder()
     */
    const sofa::helper::vector <PointID>& getPointsOnBorder();


    bool hasBorderElementLists() const;


    /** \brief: Create element lists which are on topology border:
     * - A vector of TriangleID @see m_trianglesOnBorder. ( I.e which have at least: one edge not adjacent
     to an other Triangle)
     * - A vector of EdgeID @see m_edgesOnBorder. (I.e which are adjacent to only one Triangle)
     * - A vector of PointID @see m_pointsOnBorder. (I.e which are part of only one Triangle)
     * To Do: For the moment use TriangleEdgeShellArray() in the container. To be moved in a mapping class
     */
    void createElementsOnBorder();


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


private:

    /// Set of triangle indices on topology border
    sofa::helper::vector <TriangleID> m_trianglesOnBorder;

    /// Set of edge indices on topology border
    sofa::helper::vector <EdgeID> m_edgesOnBorder;

    /// Set of point indices on topology border
    sofa::helper::vector <PointID> m_pointsOnBorder;


    /** \brief Returns a non-const triangle vertex shell given a vertex index for subsequent modification
     *
     */
    sofa::helper::vector <TriangleID>& getTriangleVertexShellForModification(const unsigned int vertexIndex);


    /** \brief Returns a non-const triangle edge shell given the index of an edge for subsequent modification
     *
     */
    sofa::helper::vector <TriangleID>& getTriangleEdgeShellForModification(const unsigned int edgeIndex);


    /** \brief Returns a non-const edge vertex shell given the index of an vertex for subsequent modification
     *
     */
    sofa::helper::vector <EdgeID>& getEdgeVertexShellForModification(const unsigned int vertexIndex);

};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
