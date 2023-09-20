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

#include <sofa/component/topology/container/dynamic/TriangleSetTopologyContainer.h>

namespace sofa::component::topology::container::dynamic
{
class TetrahedronSetTopologyModifier;

/** a class that stores a set of tetrahedra and provides access with adjacent triangles, edges and vertices */
class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API TetrahedronSetTopologyContainer : public TriangleSetTopologyContainer
{
    friend class TetrahedronSetTopologyModifier;

public:
    SOFA_CLASS(TetrahedronSetTopologyContainer,TriangleSetTopologyContainer);


    typedef core::topology::BaseMeshTopology::PointID                     PointID;
    typedef core::topology::BaseMeshTopology::EdgeID                      EdgeID;
    typedef core::topology::BaseMeshTopology::TriangleID                  TriangleID;
    typedef core::topology::BaseMeshTopology::TetraID                     TetraID;
    typedef core::topology::BaseMeshTopology::TetrahedronID               TetrahedronID;
    typedef core::topology::BaseMeshTopology::Edge                        Edge;
    typedef core::topology::BaseMeshTopology::Triangle                    Triangle;
    typedef core::topology::BaseMeshTopology::Tetra                       Tetra;
    typedef core::topology::BaseMeshTopology::SeqTetrahedra               SeqTetrahedra;
    typedef core::topology::BaseMeshTopology::TetrahedraAroundVertex      TetrahedraAroundVertex;
    typedef core::topology::BaseMeshTopology::TetrahedraAroundEdge        TetrahedraAroundEdge;
    typedef core::topology::BaseMeshTopology::TetrahedraAroundTriangle    TetrahedraAroundTriangle;
    typedef core::topology::BaseMeshTopology::EdgesInTetrahedron          EdgesInTetrahedron;
    typedef core::topology::BaseMeshTopology::TrianglesInTetrahedron      TrianglesInTetrahedron;


    typedef Tetra            Tetrahedron;
    typedef sofa::type::vector<TetraID>         VecTetraID;

protected:
    TetrahedronSetTopologyContainer();

    ~TetrahedronSetTopologyContainer() override {}
public:
    void init() override;

    //add removed tetrahedron index
    void addRemovedTetraIndex(sofa::type::vector< TetrahedronID >& tetrahedra);

    //get removed tetrahedron index
    sofa::type::vector< TetrahedronID >& getRemovedTetraIndex();

    /// Procedural creation methods
    /// @{
    void clear() override;
    void addTriangle(Index, Index, Index) override {}
    void addTetra(Index a, Index b, Index c, Index d ) override;
    /// @}



    /// BaseMeshTopology API
    /// @{

    /** \brief Returns the tetrahedra array. */
    const SeqTetrahedra& getTetrahedra() override
    {
        return getTetrahedronArray();
    }


    /** \brief Returns the tetrahedron corresponding to the TetraID i.
     *
     * @param ID of a tetrahedron.
     * @return The corresponding tetrahderon.
     */
    const Tetrahedron getTetrahedron (TetraID i) override;


    /** \brief Returns the indices of a tetrahedron given four vertex indices.
     *
     * @param the four vertex indices.
     * @return the ID of the corresponding tetrahedron.
     * @return InvalidID if none
     */
    TetrahedronID getTetrahedronIndex(PointID v1, PointID v2, PointID v3, PointID v4) override;


    /** \brief Returns the 6 edges adjacent to a given tetrahedron.
     *
     * @param ID of a tetrahedron.
     * @return EdgesInTetrahedron list composing the input tetrahedron.
     */
    const EdgesInTetrahedron& getEdgesInTetrahedron(TetraID id) override;


    /** \brief Returns the 4 triangles adjacent to a given tetrahedron.
     *
     * @param ID of a tetrahedron.
     * @return TrianglesInTetrahedron list composing the input tetrahedron.
     */
    const TrianglesInTetrahedron& getTrianglesInTetrahedron(TetraID id) override;


    /** \brief Returns the set of tetrahedra adjacent to a given vertex.
     *
     * @param ID of a vertex.
     * @return TetrahedraAroundVertex list around the input vertex.
     */
    const TetrahedraAroundVertex& getTetrahedraAroundVertex(PointID id) override;


    /** \brief Returns the set of tetrahedra adjacent to a given edge.
     *
     * @param ID of an edge.
     * @return TetrahedraAroundVertex list around the input edge.
     */
    const TetrahedraAroundEdge& getTetrahedraAroundEdge(EdgeID id) override;


    /** \brief Returns the set of tetrahedra adjacent to a given triangle.
     *
     * @param ID of a triangle.
     * @return TetrahedraAroundVertex list around the input triangle.
     */
    const TetrahedraAroundTriangle& getTetrahedraAroundTriangle(TriangleID id) override;


    /** \brief Returns the index (either 0, 1 ,2 or 3) of the vertex whose global index is vertexIndex.
     *
     * @param Ref to a Tetrahedron.
     * @param Id of a vertex.
     * @return the position of this vertex in the tetrahedron (i.e. either 0, 1, 2 or 3).
     * @return -1 if none.
     */
    int getVertexIndexInTetrahedron(const Tetrahedron &t, PointID vertexIndex) const override;


    /** \brief Returns the index (either 0, 1 ,2, 3, 4 or 5) of the edge whose global index is edgeIndex.
     *
     * @param Ref to an EdgesInTetrahedron.
     * @param Id of an edge.
     * @return the position of this edge in the tetrahedron (i.e. either 0, 1, 2, 3, 4 or 5).
     * @return -1 if none.
     */
    int getEdgeIndexInTetrahedron(const EdgesInTetrahedron &t, EdgeID edgeIndex) const override;


    /** \brief Returns the index (either 0, 1 ,2 or 3) of the triangle whose global index is triangleIndex.
     *
     * @param Ref to a TrianglesInTetrahedron.
     * @param Id of a triangle.
     * @return the position of this triangle in the tetrahedron (i.e. either 0, 1, 2 or 3).
     * @return -1 if none.
     */
    int getTriangleIndexInTetrahedron(const TrianglesInTetrahedron &t, TriangleID triangleIndex) const override;


    /** \brief Returns for each index (between 0 and 5) the two vertex indices that are adjacent to that edge.
     *
     */
    Edge getLocalEdgesInTetrahedron (const EdgeID i) const override;


    /** \brief Returns for each index (between 0 and 3) the three local vertices indices that are adjacent to that triangle
     *
     */
    Triangle getLocalTrianglesInTetrahedron (const TriangleID i) const override;

    /// @}



    /// Dynamic Topology API
    /// @{

    /// Method called by component Init method. Will create all the topology neighboorhood buffers and call @see TriangleSetTopologyContainer::initTopology()
    void initTopology();

    /** \brief Checks if the topology is coherent
     *
     * Check if the shell arrays are coherent
     * @see m_tetrahedron
     * @see m_edgesInTetrahedron
     * @see m_trianglesInTetrahedron
     * @see m_tetrahedraAroundVertex
     * @see m_tetrahedraAroundEdge
     * @see m_tetrahedraAroundTriangle
     */
    bool checkTopology() const override;


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
    const VecTetraID getConnectedElement(TetraID elem) override;

    /// Returns the set of element indices adjacent to a given element (i.e. sharing a link)
    const VecTetraID getElementAroundElement(TetraID elem) override;
    /// Returns the set of element indices adjacent to a given list of elements (i.e. sharing a link)
    const VecTetraID getElementAroundElements(VecTetraID elems) override;

    /// Returns the set of element indices adjacent to a given element with direct link from n-1 order element type (i.e triangle for tetrahedron)
    const VecTetraID getOppositeElement(TetraID elemID);
    /// @}



    /** \brief Returns the number of tetrahedra in this topology.
     *    The difference to getNbTetrahedra() is that this method does not generate the tetra array if it does not exist.
     */
    Size getNumberOfTetrahedra() const;

    /** \brief Returns the number of topological element of the current topology.
     * This function avoids to know which topological container is in used.
     */
    Size getNumberOfElements() const override;


    /** \brief Returns the Tetrahedron array. */
    const sofa::type::vector<Tetrahedron> &getTetrahedronArray();


    /** \brief Returns the EdgesInTetrahedron array (i.e. provide the 6 edge indices for each tetrahedron). */
    const sofa::type::vector< EdgesInTetrahedron > &getEdgesInTetrahedronArray() ;


    /** \brief Returns the TrianglesInTetrahedron array (i.e. provide the 4 triangle indices for each tetrahedron). */
    const sofa::type::vector< TrianglesInTetrahedron > &getTrianglesInTetrahedronArray() ;


    /** \brief Returns the TetrahedraAroundVertex array (i.e. provide the tetrahedron indices adjacent to each vertex). */
    const sofa::type::vector< TetrahedraAroundVertex > &getTetrahedraAroundVertexArray() ;


    /** \brief Returns the TetrahedraAroundEdge array (i.e. provide the tetrahedron indices adjacent to each edge). */
    const sofa::type::vector< TetrahedraAroundEdge > &getTetrahedraAroundEdgeArray() ;


    /** \brief Returns the TetrahedraAroundTriangle array (i.e. provide the tetrahedron indices adjacent to each triangle). */
    const sofa::type::vector< TetrahedraAroundTriangle > &getTetrahedraAroundTriangleArray() ;


    bool hasTetrahedra() const;

    bool hasEdgesInTetrahedron() const;

    bool hasTrianglesInTetrahedron() const;

    bool hasTetrahedraAroundVertex() const;

    bool hasTetrahedraAroundEdge() const;

    bool hasTetrahedraAroundTriangle() const;

    /// @}

    /** \brief Returns the type of the topology */
    sofa::geometry::ElementType getTopologyType() const override {return sofa::geometry::ElementType::TETRAHEDRON;}

    bool linkTopologyHandlerToData(core::topology::TopologyHandler* topologyHandler, sofa::geometry::ElementType elementType) override;

    bool unlinkTopologyHandlerToData(core::topology::TopologyHandler* topologyHandler, sofa::geometry::ElementType elementType) override;

    friend std::ostream& operator<< (std::ostream& out, const TetrahedronSetTopologyContainer& t);
    friend std::istream& operator>>(std::istream& in, TetrahedronSetTopologyContainer& t);

protected:
    /** \brief Creates the EdgeSet array.
     *
     * Create the set of edges when needed.
     */
    void createEdgeSetArray() override;


    /** \brief Creates the TriangleSet array.
     *
     * Create the array of Triangles
     */
    void createTriangleSetArray() override;


    /** \brief Creates the TetrahedronSet array.
     *
     * This function must be implemented by derived classes to create a list of tetrahedron.
     */
    virtual void createTetrahedronSetArray();


    /** \brief Creates the array of edge indices for each tetrahedron.
     *
     * This function is only called if the EdgesInTetrahedrone array is required.
     * m_edgesInTetrahedron[i] contains the 6 indices of the 6 edges of each tetrahedron
     The number of each edge is the following : edge 0 links vertex 0 and 1, edge 1 links vertex 0 and 2,
     edge 2 links vertex 0 and 3, edge 3 links vertex 1 and 2, edge 4 links vertex 1 and 3,
     edge 5 links vertex 2 and 3
    */
    virtual void createEdgesInTetrahedronArray();


    /** \brief Creates the array of triangle indices for each tetrahedron.
     *
     * This function is only called if the TrianglesInTetrahedron array is required.
     * m_trianglesInTetrahedron[i] contains the 4 indices of the 4 triangles composing the ith tetrahedron.
     */
    virtual void createTrianglesInTetrahedronArray();


    /** \brief Creates the TetrahedraAroundVertex Array.
     *
     * This function is only called if the TetrahedraAroundVertex array is required.
     * m_tetrahedraAroundVertex[i] contains the indices of all tetrahedra adjacent to the ith vertex.
     */
    virtual void createTetrahedraAroundVertexArray();


    /** \brief Creates the TetrahedraAroundEdge Array.
     *
     * This function is only called if the TetrahedraAroundEdge array is required.
     * m_tetrahedraAroundEdge[i] contains the indices of all tetrahedra adjacent to the ith edge.
     */
    virtual void createTetrahedraAroundEdgeArray();


    /** \brief Creates the TetrahedraAroundTriangle Array.
     *
     * This function is only called if the TetrahedraAroundTriangle array is required.
     * m_tetrahedraAroundTriangle[i] contains the indices of all tetrahedra adjacent to the ith triangle.
     */
    virtual void createTetrahedraAroundTriangleArray();


    void clearTetrahedra();

    void clearEdgesInTetrahedron();

    void clearTrianglesInTetrahedron();

    void clearTetrahedraAroundVertex();

    void clearTetrahedraAroundEdge();

    void clearTetrahedraAroundTriangle();


protected:

    /** \brief Returns a non-const list of tetrahedron indices around a given DOF for subsequent modification.
     *
     * @return TetrahedraAroundVertex lists in non-const.
     * @see getTetrahedraAroundVertex()
     */
    virtual TetrahedraAroundVertex& getTetrahedraAroundVertexForModification(const PointID vertexIndex);


    /** \brief Returns a non-const list of tetrahedron indices around a given edge for subsequent modification.
     *
     * @return TetrahedraAroundEdge lists in non-const.
     * @see getTetrahedraAroundEdge()
     */
    virtual TetrahedraAroundEdge& getTetrahedraAroundEdgeForModification(const EdgeID edgeIndex);


    /** \brief Returns a non-const list of tetrahedron indices around a given triangle for subsequent modification.
     *
     * @return TetrahedraAroundTriangle lists in non-const.
     * @see getTetrahedraAroundTriangle()
     */
    virtual TetrahedraAroundTriangle& getTetrahedraAroundTriangleForModification(const TriangleID triangleIndex);

    /// Use a specific boolean @see m_tetrahedronTopologyDirty in order to know if topology Data is dirty or not.
    /// Set/Get function access to this boolean
    void setTetrahedronTopologyToDirty();
    void cleanTetrahedronTopologyFromDirty();
    const bool& isTetrahedronTopologyDirty() {return m_tetrahedronTopologyDirty;}

public:
    /// force the creation of triangles
    Data<bool>  d_createTriangleArray;

    /// provides the set of tetrahedra.
    Data< sofa::type::vector<Tetrahedron> > d_tetrahedron;
protected:
    /// provides the set of edges for each tetrahedron.
    sofa::type::vector<EdgesInTetrahedron> m_edgesInTetrahedron;

    /// provides the set of triangles for each tetrahedron.
    sofa::type::vector<TrianglesInTetrahedron> m_trianglesInTetrahedron;

    /// for each vertex provides the set of tetrahedra adjacent to that vertex.
    sofa::type::vector< TetrahedraAroundVertex > m_tetrahedraAroundVertex;

    /// for each edge provides the set of tetrahedra adjacent to that edge.
    sofa::type::vector< TetrahedraAroundEdge > m_tetrahedraAroundEdge;

    /// removed tetrahedron index
    sofa::type::vector<TetrahedronID> m_removedTetraIndex;

    /// for each triangle provides the set of tetrahedra adjacent to that triangle.
    sofa::type::vector< TetrahedraAroundTriangle > m_tetrahedraAroundTriangle;


    /// Boolean used to know if the topology Data of this container is dirty
    bool m_tetrahedronTopologyDirty = false;
};

} //namespace sofa::component::topology::container::dynamic
