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
#pragma once
#include <sofa/component/topology/container/dynamic/config.h>

#include <sofa/component/topology/container/dynamic/EdgeSetTopologyContainer.h>

namespace sofa::component::topology::container::dynamic
{
class TriangleSetTopologyModifier;


/*! \class TriangleSetTopologyContainer
\brief: Object that stores a set of triangles and provides access
to each triangle and its edges and vertices */
class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API TriangleSetTopologyContainer : public EdgeSetTopologyContainer
{
    friend class TriangleSetTopologyModifier;

public:
    SOFA_CLASS(TriangleSetTopologyContainer,EdgeSetTopologyContainer);

    typedef core::topology::BaseMeshTopology::PointID                      PointID;
    typedef core::topology::BaseMeshTopology::EdgeID                       EdgeID;
    typedef core::topology::BaseMeshTopology::TriangleID                   TriangleID;
    typedef core::topology::BaseMeshTopology::Edge                         Edge;
    typedef core::topology::BaseMeshTopology::Triangle                     Triangle;
    typedef core::topology::BaseMeshTopology::SeqTriangles                 SeqTriangles;
    typedef core::topology::BaseMeshTopology::EdgesInTriangle              EdgesInTriangle;
    typedef core::topology::BaseMeshTopology::TrianglesAroundVertex        TrianglesAroundVertex;
    typedef core::topology::BaseMeshTopology::TrianglesAroundEdge          TrianglesAroundEdge;
    typedef sofa::type::vector<TriangleID>                               VecTriangleID;


protected:
    TriangleSetTopologyContainer();

    ~TriangleSetTopologyContainer() override {}
public:
    void init() override;

    void reinit() override;


    /// Procedural creation methods
    /// @{
    void clear() override;
    void addEdge(Index, Index) override {}
    void addTriangle(Index a, Index b, Index c ) override;
    /// @}

    /// BaseMeshTopology API
    /// @{

    /** \brief Returns the quad array. */
    const SeqTriangles& getTriangles() override
    {
        return getTriangleArray();
    }


    /** \brief Returns the triangle corresponding to the TriangleID i.
     *
     * @param ID of a triangle.
     * @return The corresponding triangle.
     */
    const Triangle getTriangle(TriangleID i) override;


    /* Returns the indices of a triangle given three vertex indices.
     *
     * @param the three vertex indices.
     * @return the ID of the corresponding triangle.
     * @return InvalidID if none
     */
    TriangleID getTriangleIndex(PointID v1, PointID v2, PointID v3) override;


    /** \brief Returns the 3 edges adjacent to a given triangle.
     *
     * @param ID of a triangle.
     * @return EdgesInTriangle list composing the input triangle.
     */
    const EdgesInTriangle& getEdgesInTriangle(TriangleID id) override;


    /** \brief Returns the set of triangles adjacent to a given vertex.
     *
     * @param ID of a vertex
     * @return TrianglesAroundVertex list around the input vertex
     */
    const TrianglesAroundVertex& getTrianglesAroundVertex(PointID id) override;


    /** \brief Returns the set of triangles adjacent to a given edge.
     *
     * @param ID of an edge.
     * @return TrianglesAroundEdge list around the input edge.
     */
    const TrianglesAroundEdge& getTrianglesAroundEdge(EdgeID id) override;


    /** \brief Returns the index (either 0, 1 ,2) of the vertex whose global index is vertexIndex.
     *
     * @param Ref to a triangle.
     * @param Id of a vertex.
     * @return the position of this vertex in the triangle (i.e. either 0, 1, 2).
     * @return -1 if none.
     */
    int getVertexIndexInTriangle(const Triangle &t, PointID vertexIndex) const override;

    /** \brief Returns the index (either 0, 1 ,2) of the edge whose global index is edgeIndex.
     *
     * @param Ref to an EdgesInTriangle.
     * @param Id of an edge.
     * @return the position of this edge in the triangle (i.e. either 0, 1, 2).
     * @return -1 if none.
     */
    int getEdgeIndexInTriangle(const EdgesInTriangle &t, EdgeID edgeIndex) const override;

    /** \brief Returns the global point index of third point of a triangle given the 2 others.
     *
     * @param Ref to an Triangle.
     * @param 2 point indices of this triangle.
     * @return third point index.
     */
    PointID getOtherPointInTriangle(const Triangle& t, PointID p1, PointID p2) const;

    /// @}



    /// Dynamic Topology API
    /// @{
    /// Method called by component Init method. Will create all the topology neighboorhood buffers and call @see EdgeSetTopologyContainer::initTopology()
    void initTopology();

    /** \brief Checks if the topology is coherent
     *
     * Check if the shell arrays are coherent
     * @see m_triangle
     * @see m_edgesInTriangle
     * @see m_trianglesAroundVertex
     * @see m_trianglesAroundEdge
     */
    bool checkTopology() const override;


    /** \brief Returns the number of triangles in this topology.
     *    The difference to getNbTriangles() is that this method does not generate the triangle array if it does not exist.
     */
    Size getNumberOfTriangles() const;

    /** \brief Returns the number of topological element of the current topology.
     * This function avoids to know which topological container is in used.
     */
    Size getNumberOfElements() const override;

    /** \brief Returns the Triangle array. */
    const sofa::type::vector<Triangle> &getTriangleArray();


    /** \brief Returns the EdgesInTriangle array (i.e. provide the 3 edge indices for each triangle). */
    const sofa::type::vector< EdgesInTriangle > &getEdgesInTriangleArray() ;


    /** \brief Returns the TrianglesAroundVertex array (i.e. provide the triangles indices adjacent to each vertex). */
    const sofa::type::vector< TrianglesAroundVertex > &getTrianglesAroundVertexArray();


    /** \brief Returns the TrianglesAroundEdge array (i.e. provide the triangles indices adjacent to each edge). */
    const sofa::type::vector< TrianglesAroundEdge > &getTrianglesAroundEdgeArray() ;


    /** \brief: Return a list of TriangleID which are on a border.
     * @see createElementsOnBorder()
     */
    const sofa::type::vector<TriangleID>& getTrianglesOnBorder() override;


    /** \brief: Return a list of EdgeID which are on a border.
     * @see createElementsOnBorder()
     */
    const sofa::type::vector<EdgeID>& getEdgesOnBorder() override;


    /** \brief: Return a vector of PointID which are on a border.
     * @see createElementsOnBorder()
     */
    const sofa::type::vector<PointID>& getPointsOnBorder() override;


    /// Get information about connexity of the mesh
    /// @{
    /** \brief Checks if the topology has only one connected component
      *
      * @return true if only one connected component
      */
    bool checkConnexity() override;

    /// Returns the number of connected component.
    Size getNumberOfConnectedComponent() override;

    /// Returns the set of element indices connected to an input one (i.e. which can be reached by topological links)
    const VecTriangleID getConnectedElement(TriangleID elem) override;

    /// Returns the set of element indices adjacent to a given element (i.e. sharing a link)
    const VecTriangleID getElementAroundElement(TriangleID elem) override;
    /// Returns the set of element indices adjacent to a given list of elements (i.e. sharing a link)
    const VecTriangleID getElementAroundElements(VecTriangleID elems) override;
    /// @}

    bool hasTriangles() const;

    bool hasEdgesInTriangle() const;

    bool hasTrianglesAroundVertex() const;

    bool hasTrianglesAroundEdge() const;

    bool hasBorderElementLists() const;


    /** \brief: Create element lists which are on topology border:
     *
     * - A vector of TriangleID @see m_trianglesOnBorder. ( I.e which have at least: one edge not adjacent
     to an other Triangle)
     * - A vector of EdgeID @see m_edgesOnBorder. (I.e which are adjacent to only one Triangle)
     * - A vector of PointID @see m_pointsOnBorder. (I.e which are part of only one Triangle)
     */
    void createElementsOnBorder();

    /// @}

    /// Will change order of vertices in triangle: t[1] <=> t[2]
    void reOrientateTriangle(TriangleID id) override;

    /** \brief Returns the type of the topology */
    sofa::geometry::ElementType getTopologyType() const override { return sofa::geometry::ElementType::TRIANGLE; }
    
    bool linkTopologyHandlerToData(core::topology::TopologyHandler* topologyHandler, sofa::geometry::ElementType elementType) override;

    bool unlinkTopologyHandlerToData(core::topology::TopologyHandler* topologyHandler, sofa::geometry::ElementType elementType) override;

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
    void createEdgeSetArray() override;


    /** \brief Creates the array of edge indices for each triangle.
     *
     * This function is only called if the EdgesInTriangle array is required.
     * m_edgesInTriangle[i] contains the 3 indices of the 3 edges composing the ith triangle.
     */
    virtual void createEdgesInTriangleArray();


    /** \brief Creates the TrianglesAroundVertex Array.
     *
     * This function is only called if the TrianglesAroundVertex array is required.
     * m_trianglesAroundVertex[i] contains the indices of all triangles adjacent to the ith DOF.
     */
    virtual void createTrianglesAroundVertexArray();


    /** \brief Creates the TrianglesAroundEdge Array.
     *
     * This function is only called if the TrianglesAroundVertex array is required.
     * m_trianglesAroundEdge[i] contains the indices of all triangles adjacent to the ith edge.
     */
    virtual void createTrianglesAroundEdgeArray();


    void clearTriangles();

    void clearEdgesInTriangle();

    void clearTrianglesAroundVertex();

    void clearTrianglesAroundEdge();

    void clearBorderElementLists();


protected:

    /** \brief Returns a non-const list of triangle indices around a given DOF for subsequent modification.
     *
     * @return TrianglesAroundVertex lists in non-const.
     * @see getTrianglesAroundVertex()
     */
    virtual TrianglesAroundVertex& getTrianglesAroundVertexForModification(const PointID vertexIndex);


    /** \brief Returns a non-const list of triangle indices around a given edge for subsequent modification.
     *
     * @return TrianglesAroundEdge lists in non-const.
     * @see getTrianglesAroundEdge()
     */
    virtual TrianglesAroundEdge& getTrianglesAroundEdgeForModification(const EdgeID edgeIndex);

    /// Use a specific boolean @see m_triangleTopologyDirty in order to know if topology Data is dirty or not.
    /// Set/Get function access to this boolean
    void setTriangleTopologyToDirty();
    void cleanTriangleTopologyFromDirty();
    const bool& isTriangleTopologyDirty() {return m_triangleTopologyDirty;}

public:
    /// provides the set of triangles.
    Data< sofa::type::vector<Triangle> > d_triangle;

protected:
    /// provides the 3 edges in each triangle.
    sofa::type::vector<EdgesInTriangle> m_edgesInTriangle;

    /// for each vertex provides the set of triangles adjacent to that vertex.
    sofa::type::vector< TrianglesAroundVertex > m_trianglesAroundVertex;

    /// for each edge provides the set of triangles adjacent to that edge.
    sofa::type::vector< TrianglesAroundEdge > m_trianglesAroundEdge;

    /// Set of triangle indices on topology border.
    sofa::type::vector<TriangleID> m_trianglesOnBorder;

    /// Set of edge indices on topology border.
    sofa::type::vector<EdgeID> m_edgesOnBorder;

    /// Set of point indices on topology border.
    sofa::type::vector<PointID> m_pointsOnBorder;

    /// Boolean used to know if the topology Data of this container is dirty
    bool m_triangleTopologyDirty = false;

};

} //namespace sofa::component::topology::container::dynamic
