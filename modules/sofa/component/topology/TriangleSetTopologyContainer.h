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
#ifndef SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYCONTAINER_H

#include <sofa/component/topology/EdgeSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace topology
{
class TriangleSetTopologyModifier;

using core::componentmodel::topology::BaseMeshTopology;

typedef BaseMeshTopology::PointID			PointID;
typedef BaseMeshTopology::EdgeID			EdgeID;
typedef BaseMeshTopology::TriangleID		TriangleID;
typedef BaseMeshTopology::Edge			Edge;
typedef BaseMeshTopology::Triangle		Triangle;
typedef BaseMeshTopology::SeqTriangles		SeqTriangles;
typedef BaseMeshTopology::EdgesInTriangle		EdgesInTriangle;
typedef BaseMeshTopology::TrianglesAroundVertex	TrianglesAroundVertex;
typedef BaseMeshTopology::TrianglesAroundEdge	TrianglesAroundEdge;

/*! \class TriangleSetTopologyContainer
\brief: Object that stores a set of triangles and provides access
to each triangle and its edges and vertices */
class SOFA_COMPONENT_CONTAINER_API TriangleSetTopologyContainer : public EdgeSetTopologyContainer
{
    friend class TriangleSetTopologyModifier;

public:

    TriangleSetTopologyContainer();

    TriangleSetTopologyContainer(const sofa::helper::vector< Triangle > &triangles );

    virtual ~TriangleSetTopologyContainer() {}

    /// Procedural creation methods
    /// @{
    virtual void clear();
    virtual void addEdge( int, int ) {}
    virtual void addTriangle( int a, int b, int c );
    /// @}

    virtual void init();

    /// BaseMeshTopology API
    /// @{
    const SeqTriangles& getTriangles()
    {
        return getTriangleArray();
    }

    /* Returns the indices of a triangle given three vertex indices : returns -1 if none */
    virtual int getTriangleIndex(PointID v1, PointID v2, PointID v3);

    /// Returns the set of edges adjacent to a given triangle.
    /*	const EdgesInTriangle& getEdgesInTriangle(TriangleID i)
    {
      return getEdgesInTriangle(i);
    }
    */
    /** \brief Returns the set of triangles adjacent to a given vertex.
     *
     */
    virtual const TrianglesAroundVertex& getTrianglesAroundVertex(PointID i);


    /** \brief Returns the set of triangles adjacent to a given edge.
     *
     */
    virtual const TrianglesAroundEdge& getTrianglesAroundEdge(EdgeID i) ;

    /** returns the index (either 0, 1 ,2) of the vertex whose global index is vertexIndex. Returns -1 if none */
    virtual int getVertexIndexInTriangle(const Triangle &t, PointID vertexIndex) const;

    /** returns the index (either 0, 1 ,2) of the edge whose global index is edgeIndex. Returns -1 if none */
    virtual int getEdgeIndexInTriangle(const EdgesInTriangle &t, EdgeID edgeIndex) const;

    /// @}

    /** \brief Checks if the topology is coherent
     *
     * Check if the shell arrays are coherent
     */
    virtual bool checkTopology() const;

    /** \brief Returns the Triangle array.
     *
     */
    const sofa::helper::vector<Triangle> &getTriangleArray();

    /** \brief Returns the Triangle Vertex Shells array.
     *
     */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getTrianglesAroundVertexArray();

    /** \brief Returns the EdgesInTriangle array (ie provide the 3 edge indices for each triangle)
     *
     */
    const sofa::helper::vector< EdgesInTriangle > &getEdgesInTriangleArray() ;

    /** \brief Returns the Triangle Edge Shells array (ie provides the triangles adjacent to each edge)
     *
     */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getTrianglesAroundEdgeArray() ;

    /** \brief Returns the number of triangles in this topology.
     *	The difference to getNbTriangles() is that this method does not generate the triangle array if it does not exist.
     */
    unsigned int getNumberOfTriangles() const;


    /** \brief Returns the 3 edges adjacent to a given triangle.
     *
     */
    virtual const EdgesInTriangle& getEdgesInTriangle(TriangleID i) ;

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
     * To Do: For the moment use TrianglesAroundEdgeArray() in the container. To be moved in a mapping class
     */
    void createElementsOnBorder();

protected:
    /** \brief Creates the TriangleSet array.
     *
     * This function must be implemented by derived classes to create a list of triangles from a set of tetrahedra for instance
     */
    virtual void createTriangleSetArray();

    /** \brief Creates the EdgeSet array.
     *
     * Create the set of edges when needed.
     */
    virtual void createEdgeSetArray();

    /** \brief Creates the array of edge indices for each triangle
     *
     * This function is only called if the EdgesInTriangle array is required.
     * m_edgesInTriangle[i] contains the 3 indices of the 3 edges opposite to the ith vertex
     */
    void createEdgesInTriangleArray();

    /** \brief Creates the Triangle Vertex Shell Array
     *
     * This function is only called if the TrianglesAroundVertex array is required.
     * m_trianglesAroundVertex[i] contains the indices of all triangles adjacent to the ith vertex
     */
    virtual void createTrianglesAroundVertexArray();

    /** \brief Creates the Triangle Edge Shell Array
     *
     * This function is only called if the TrianglesAroundVertex array is required.
     * m_trianglesAroundEdge[i] contains the indices of all triangles adjacent to the ith edge
     */
    virtual void createTrianglesAroundEdgeArray();


    bool hasTriangles() const;

    bool hasEdgesInTriangle() const;

    bool hasTrianglesAroundVertex() const;

    bool hasTrianglesAroundEdge() const;

    void clearTriangles();

    void clearEdgesInTriangle();

    void clearTrianglesAroundVertex();

    void clearTrianglesAroundEdge();

private:
    /** \brief Returns a non-const triangle vertex shell given a vertex index for subsequent modification
     *
     */
    sofa::helper::vector< unsigned int > &getTrianglesAroundVertexForModification(const unsigned int vertexIndex);
    /** \brief Returns a non-const triangle edge shell given the index of an edge for subsequent modification
     *
     */
    sofa::helper::vector< unsigned int > &getTrianglesAroundEdgeForModification(const unsigned int edgeIndex);

protected:
    /// provides the set of triangles
    sofa::helper::vector<Triangle> m_triangle;
    DataPtr< sofa::helper::vector<Triangle> > d_triangle;
    /// provides the 3 edges in each triangle
    sofa::helper::vector<EdgesInTriangle> m_edgesInTriangle;
    /// for each vertex provides the set of triangles adjacent to that vertex
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_trianglesAroundVertex;
    /// for each edge provides the set of triangles adjacent to that edge
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_trianglesAroundEdge;

    /// Set of triangle indices on topology border
    sofa::helper::vector <TriangleID> m_trianglesOnBorder;

    /// Set of edge indices on topology border
    sofa::helper::vector <EdgeID> m_edgesOnBorder;

    /// Set of point indices on topology border
    sofa::helper::vector <PointID> m_pointsOnBorder;


    virtual void loadFromMeshLoader(sofa::component::container::MeshLoader* loader);
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
