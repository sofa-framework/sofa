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
#ifndef SOFA_COMPONENT_TOPOLOGY_MANIFOLDTRIANGLESETTOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_MANIFOLDTRIANGLESETTOPOLOGYCONTAINER_H
#include <ManifoldTopologies/config.h>

#include <ManifoldTopologies/config.h>

#include <SofaBaseTopology/TriangleSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace topology
{

using core::topology::BaseMeshTopology;
typedef BaseMeshTopology::PointID	        	PointID;
typedef BaseMeshTopology::EdgeID	        	EdgeID;
typedef BaseMeshTopology::TriangleID      	TriangleID;
typedef BaseMeshTopology::Edge	           	Edge;
typedef BaseMeshTopology::Triangle        	Triangle;
typedef BaseMeshTopology::SeqTriangles       	SeqTriangles;
typedef BaseMeshTopology::EdgesInTriangle      	EdgesInTriangle;
typedef BaseMeshTopology::EdgesAroundVertex      	EdgesAroundVertex;
typedef BaseMeshTopology::TrianglesAroundVertex	TrianglesAroundVertex;
typedef BaseMeshTopology::TrianglesAroundEdge	TrianglesAroundEdge;

/** A class that stores a set of triangles and provides access
 to each triangle, triangle's edges and vertices.
 This topology is contraint by the manifold property: each edge is adjacent to either one or at most two Triangles.*/
class SOFA_MANIFOLD_TOPOLOGIES_API ManifoldTriangleSetTopologyContainer : public TriangleSetTopologyContainer
{

    friend class ManifoldTriangleSetTopologyModifier;

public:
    SOFA_CLASS(ManifoldTriangleSetTopologyContainer,TriangleSetTopologyContainer);

    ManifoldTriangleSetTopologyContainer();

    virtual ~ManifoldTriangleSetTopologyContainer() {}

    virtual void init();

    virtual void clear();



    /** \brief Checks if the topology is coherent.
     *
     * The function TriangleSetTopologyContainer::CheckTopology() Check if the shell arrays are coherent
     * from on to the other.
     * In this class, we check the topology. I.e for m_trianglesAroundVertex
     *   - Test if triangles are stocked in counterclockewise direction around each vertex and that they are contiguous.
     *   - Test if no triangles are missing and if no triangle are badly connected to a vertex.
     * For m_trianglesAroundEdge
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
    int getNextTrianglesAroundVertex(PointID vertexIndex, TriangleID triangleIndex);


    /** \brief: Given a Triangle and a Vertex i, returns the next adjacent triangle to this first one
     * in the clockwise direction around the ith vertex.
     *
     * @return -1 if there is no adjacent triangle in this direction
     * @return -2 if the vertex does not belongs to this Triangle or if there is an other error.
     */
    int getPreviousTrianglesAroundVertex(PointID vertexIndex, TriangleID triangleIndex);


    /** \brief: Given a Triangle and a Edge i, returns the other adjacent triangle to the ith edge.
     *
     * @return -1 if there is only one triangle adjacent to this edge.
     * @return -2 if the edge does not belongs to this Triangle or if there is an other error.
     */
    int getOppositeTrianglesAroundEdge(EdgeID edgeIndex, TriangleID triangleIndex);


    /** \brief: Given a Edge and a Vertex i, returns the next edge containing the ith vertex
     * in the counterclockwise direction around the ith vertex.
     *
     * return -1 if there is adjacent no triangle in this direction
     * return -2 if the vertex does not belongs to the edge or if there is an other error.
     */
    int getNextEdgesAroundVertex(PointID vertexIndex, EdgeID edgeIndex);


    /** \brief: Given a Edge and a Vertex i, returns the next edge containing the ith vertex
     * in the clockwise direction around the ith vertex.
     *
     * return -1 if there is no triangle in this direction
     * return -2 if the vertex does not belongs to the edge or if there is an other error.
     */
    int getPreviousEdgesAroundVertex(PointID vertexIndex, EdgeID edgeIndex);

    /** \brief return the orientation of an edge relatively to one triangle
     *
     * @param Ref to the triangle of reference
     * @parem Ref to the edge to test.
     * @return 1 if positive orientation
     * @return -1 if negative orientation
     * @return 0 if edge doesn't belongs to this triangle
     */
    int getEdgeTriangleOrientation(const Triangle& f, const Edge& e);

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
     * This function is only called if the EdgesAroundVertex array is required.
     * m_EdgesAroundVertex[i] contains the indices of all edges adjacent to the ith vertex.
     * This funciton check if there are T connections between more than 2 edges at the ith DOF.
     *
     */
    virtual void createEdgesAroundVertexArray();


    /** \brief Creates the Triangle Vertex Shell Array
     *
     * This function is only called if the TrianglesAroundVertex array is required.
     * m_trianglesAroundVertex[i] contains the indices of all triangles adjacent to the ith vertex.
     * This function check if there are T connections between more than 3 triangles at the ith DOF.
     *
     */
    virtual void createTrianglesAroundVertexArray();


    /** \brief Creates the Triangle Edge Shell Array
     *
     * This function is only called if the TrianglesAroundEdge array is required.
     * m_trianglesAroundEdge[i] contains the indices of all triangles adjacent to the ith edge.
     * This function check if there are more than 2 triangles adjacent to each edge.
     *
     */
    virtual void createTrianglesAroundEdgeArray();


private:

    /** \brief Returns a non-const triangle vertex shell given a vertex index for subsequent modification
     *
     */
    virtual TrianglesAroundVertex& getTrianglesAroundVertexForModification(const PointID vertexIndex);


    /** \brief Returns a non-const triangle edge shell given the index of an edge for subsequent modification
     *
     */
    virtual TrianglesAroundEdge& getTrianglesAroundEdgeForModification(const EdgeID edgeIndex);


    /** \brief Returns a non-const edge vertex shell given the index of an vertex for subsequent modification
     *
     */
    virtual EdgesAroundVertex& getEdgesAroundVertexForModification(const PointID vertexIndex);

};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
