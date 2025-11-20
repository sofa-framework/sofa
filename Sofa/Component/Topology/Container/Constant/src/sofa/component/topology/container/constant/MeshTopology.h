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
#include <sofa/component/topology/container/constant/config.h>

#include <sofa/core/topology/Topology.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/DataEngine.h>
#include <sofa/type/vector.h>

namespace sofa::component::topology::container::constant
{

class SOFA_COMPONENT_TOPOLOGY_CONTAINER_CONSTANT_API MeshTopology : public core::topology::BaseMeshTopology
{
public:
    SOFA_CLASS(MeshTopology,core::topology::BaseMeshTopology);
protected:

    class PrimitiveUpdate : public sofa::core::DataEngine
    {
    public:
        typedef Topology::Edge Edge;
        typedef Topology::Quad Quad;
        typedef Topology::Triangle Triangle;
        typedef Topology::Hexa Hexa;
        typedef Topology::Tetra Tetra;
        SOFA_ABSTRACT_CLASS(PrimitiveUpdate,sofa::core::DataEngine);
        PrimitiveUpdate(MeshTopology* t):topology(t) {}
    protected:
        MeshTopology* topology;
    };
private:

    class EdgeUpdate : public PrimitiveUpdate
    {
    public:
        SOFA_CLASS(EdgeUpdate,PrimitiveUpdate);
        EdgeUpdate(MeshTopology* t);
        void doUpdate() override;
    protected:
        void updateFromVolume();
        void updateFromSurface();
    };


    class TriangleUpdate : public PrimitiveUpdate
    {
    public:

        SOFA_CLASS(TriangleUpdate,PrimitiveUpdate);
        TriangleUpdate(MeshTopology* t);
        void doUpdate() override;
    };

    class QuadUpdate : public PrimitiveUpdate
    {
    public:
        SOFA_CLASS(QuadUpdate,PrimitiveUpdate);
        QuadUpdate(MeshTopology* t);
        void doUpdate() override;
    };
protected:
    MeshTopology();
public:
    void init() override;

    /// Method called by component Init method. Will create all the topology buffers
    void computeCrossElementBuffers() override;

    Size getNbPoints() const override;

    void setNbPoints(Size  n) override;

    // Complete sequence accessors

    const SeqEdges& getEdges() override;
    const SeqTriangles& getTriangles() override;
    const SeqQuads& getQuads() override;
    const SeqTetrahedra& getTetrahedra() override;
    const SeqHexahedra& getHexahedra() override;
    const SeqPrisms& getPrisms() override;
    const SeqPyramids& getPyramids() override;

    // If using STEP loader, include also uv coordinates
    typedef Index					UVID;
    typedef type::Vec2						UV;
    typedef type::vector<UV>				SeqUV;
    virtual const SeqUV& getUVs();
    virtual Size getNbUVs();
    virtual const UV getUV(UVID i);
    void addUV(SReal u, SReal v);
    //

    /// @name neighbors queries for Edge Topology
    /// @{
    /// Returns the set of edges adjacent to a given vertex.
    const EdgesAroundVertex &getEdgesAroundVertex(PointID i) override;
    /** \brief Returns the TrianglesAroundVertex array (i.e. provide the triangles indices adjacent to each vertex). */
    const type::vector< EdgesAroundVertex > &getEdgesAroundVertexArray();
    /// @}


    /// @name neighbors queries for Triangle Topology
    /// @{
    /// Returns the set of triangle adjacent to a given vertex.
    const TrianglesAroundVertex &getTrianglesAroundVertex(PointID i) override;
    /** \brief Returns the TrianglesAroundVertex array (i.e. provide the triangles indices adjacent to each vertex). */
    const type::vector< TrianglesAroundVertex > &getTrianglesAroundVertexArray();

    /// Returns the set of 3 edge indices of a given triangle.
    const EdgesInTriangle &getEdgesInTriangle(TriangleID i) override;
    /** \brief Returns the EdgesInTriangle array (i.e. provide the 3 edge indices for each triangle). */
    const type::vector< EdgesInTriangle > &getEdgesInTriangleArray();
    /// Returns the set of triangle adjacent to a given edge.
    const TrianglesAroundEdge &getTrianglesAroundEdge(EdgeID i) override;
    /** \brief Returns the TrianglesAroundEdge array (i.e. provide the triangles indices adjacent to each edge). */
    const type::vector< TrianglesAroundEdge > &getTrianglesAroundEdgeArray();
    /// @}


    /// @name neighbors queries for Quad Topology
    /// @{
    /// Returns the set of quad adjacent to a given vertex.
    const QuadsAroundVertex &getQuadsAroundVertex(PointID i) override;
    /** \brief Returns the QuadsAroundVertex array (i.e. provide the quad indices adjacent to each vertex). */
    const type::vector< QuadsAroundVertex > &getQuadsAroundVertexArray();

    /// Returns the set of edges adjacent to a given quad.
    const EdgesInQuad &getEdgesInQuad(QuadID i) override;
    /** \brief Returns the EdgesInQuadArray array (i.e. provide the 4 edge indices for each quad) */
    const type::vector< EdgesInQuad > &getEdgesInQuadArray();
    /// Returns the set of quad adjacent to a given edge.
    const QuadsAroundEdge &getQuadsAroundEdge(EdgeID i) override;
    /** \brief Returns the QuadsAroundEdge array (i.e. provide the quad indices adjacent to each edge). */
    const type::vector< QuadsAroundEdge > &getQuadsAroundEdgeArray();
    /// @}


    /// @name neighbors queries for Tetrahedron Topology
    /// @{
    /// Returns the set of tetrahedra adjacent to a given vertex.
    const TetrahedraAroundVertex& getTetrahedraAroundVertex(PointID i) override;
    /** \brief Returns the TetrahedraAroundVertex array (i.e. provide the tetrahedron indices adjacent to each vertex). */
    const type::vector< TetrahedraAroundVertex > &getTetrahedraAroundVertexArray();

    /// Returns the set of edges adjacent to a given tetrahedron.
    const EdgesInTetrahedron& getEdgesInTetrahedron(TetraID i) override;
    /** \brief Returns the EdgesInTetrahedron array (i.e. provide the 6 edge indices for each tetrahedron). */
    const type::vector< EdgesInTetrahedron > &getEdgesInTetrahedronArray();
    /// Returns the set of tetrahedra adjacent to a given edge.
    const TetrahedraAroundEdge& getTetrahedraAroundEdge(EdgeID i) override;
    /** \brief Returns the TetrahedraAroundEdge array (i.e. provide the tetrahedron indices adjacent to each edge). */
    const type::vector< TetrahedraAroundEdge > &getTetrahedraAroundEdgeArray();

    /// Returns the set of triangles adjacent to a given tetrahedron.
    const TrianglesInTetrahedron& getTrianglesInTetrahedron(TetraID i) override;
    /** \brief Returns the TrianglesInTetrahedron array (i.e. provide the 4 triangle indices for each tetrahedron). */
    const type::vector< TrianglesInTetrahedron > &getTrianglesInTetrahedronArray();
    /// Returns the set of tetrahedra adjacent to a given triangle.
    const TetrahedraAroundTriangle& getTetrahedraAroundTriangle(TriangleID i) override;
    /** \brief Returns the TetrahedraAroundTriangle array (i.e. provide the tetrahedron indices adjacent to each triangle). */
    const type::vector< TetrahedraAroundTriangle > &getTetrahedraAroundTriangleArray();
    /// @}


    /// @name neighbors queries for Hexhaedron Topology
    /// @{
    /// Returns the set of hexahedra adjacent to a given vertex.
    const HexahedraAroundVertex& getHexahedraAroundVertex(PointID i) override;
    /** \brief Returns the HexahedraAroundVertex array (i.e. provide the hexahedron indices adjacent to each vertex).*/
    const type::vector< HexahedraAroundVertex > &getHexahedraAroundVertexArray();

    /// Returns the set of edges adjacent to a given hexahedron.
    const EdgesInHexahedron& getEdgesInHexahedron(HexaID i) override;
    /** \brief Returns the EdgesInHexahedron array (i.e. provide the 12 edge indices for each hexahedron).	*/
    const type::vector< EdgesInHexahedron > &getEdgesInHexahedronArray();
    /// Returns the set of hexahedra adjacent to a given edge.
    const HexahedraAroundEdge& getHexahedraAroundEdge(EdgeID i) override;
    /** \brief Returns the HexahedraAroundEdge array (i.e. provide the hexahedron indices adjacent to each edge). */
    const type::vector< HexahedraAroundEdge > &getHexahedraAroundEdgeArray();

    /// Returns the set of quads adjacent to a given hexahedron.
    const QuadsInHexahedron& getQuadsInHexahedron(HexaID i) override;
    /** \brief Returns the QuadsInHexahedron array (i.e. provide the 8 quad indices for each hexahedron).	*/
    const type::vector< QuadsInHexahedron > &getQuadsInHexahedronArray();
    /// Returns the set of hexahedra adjacent to a given quad.
    const HexahedraAroundQuad& getHexahedraAroundQuad(QuadID i) override;
    /** \brief Returns the HexahedraAroundQuad array (i.e. provide the hexahedron indices adjacent to each quad). */
    const type::vector< HexahedraAroundQuad > &getHexahedraAroundQuadArray();
    /// @}


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
    virtual const type::vector<Index> getConnectedElement(Index elem) override;

    /// Returns the set of element indices adjacent to a given element (i.e. sharing a link)
    virtual const type::vector<Index> getElementAroundElement(Index elem) override;
    /// Returns the set of element indices adjacent to a given list of elements (i.e. sharing a link)
    virtual const type::vector<Index> getElementAroundElements(type::vector<Index> elems) override;
    /// @}

    // Get point positions (same methods as points accessors but not inherited)
    SReal getPosX(Index i) const;
    SReal getPosY(Index i) const;
    SReal getPosZ(Index i) const;

    // Points accessors (not always available)

    bool hasPos() const override;
    SReal getPX(Index i) const override;
    SReal getPY(Index i) const override;
    SReal getPZ(Index i) const override;

    // for procedural creation without file loader
    void clear() override;
    void addPoint(SReal px, SReal py, SReal pz) override;
    void addEdge( Index a, Index b ) override;
    void addTriangle( Index a, Index b, Index c ) override;
    void addQuad( Index a, Index b, Index c, Index d ) override;
    void addTetra( Index a, Index b, Index c, Index d ) override;
    void addHexa( Index a, Index b, Index c, Index d, Index e, Index f, Index g, Index h ) override;

    /// get the current revision of this mesh (use to detect changes)
    int getRevision() const override { return revision; }


    void draw(const core::visual::VisualParams* vparams) override;

    virtual bool hasVolume() { return ( ( getNbTetrahedra() + getNbHexahedra() ) > 0 ); }
    virtual bool hasSurface() { return ( ( getNbTriangles() + getNbQuads() ) > 0 ); }
    virtual bool hasLines() { return ( ( getNbLines() ) > 0 ); }

    virtual bool isVolume() { return hasVolume(); }
    virtual bool isSurface() { return !hasVolume() && hasSurface(); }
    virtual bool isLines() { return !hasVolume() && !hasSurface() && hasLines(); }


    /// Returns the set of edges adjacent to a given vertex.
    virtual const EdgesAroundVertex &getOrientedEdgesAroundVertex(PointID i);

    /// Returns the set of oriented triangle adjacent to a given vertex.
    virtual const TrianglesAroundVertex &getOrientedTrianglesAroundVertex(PointID i);

    /// Returns the set of oriented quad adjacent to a given vertex.
    virtual const QuadsAroundVertex &getOrientedQuadsAroundVertex(PointID i);


    // test whether p0p1 has the same orientation as triangle t
    // opposite direction: return -1
    // same direction: return 1
    // otherwise: return 0
    int computeRelativeOrientationInTri(const PointID ind_p0, const PointID ind_p1, const PointID ind_t);

    // test whether p0p1 has the same orientation as triangle t
    // opposite direction: return -1
    // same direction: return 1
    // otherwise: return 0
    int computeRelativeOrientationInQuad(const PointID ind_p0, const PointID ind_p1, const PointID ind_q);

    /// Will change order of vertices in triangle: t[1] <=> t[2]
    void reOrientateTriangle(TriangleID id) override;

public:
    typedef type::vector<type::Vec3> SeqPoints;

    Data< SeqPoints > d_seqPoints; ///< List of point positions
    Data<SeqEdges> d_seqEdges; ///< List of edge indices
    Data<SeqTriangles> d_seqTriangles; ///< List of triangle indices
    Data<SeqQuads>       d_seqQuads; ///< List of quad indices
    Data<SeqTetrahedra>      d_seqTetrahedra; ///< List of tetrahedron indices
    Data<SeqHexahedra>	   d_seqHexahedra; ///< List of hexahedron indices
    Data<SeqPrisms> d_seqPrisms;
    Data<SeqPyramids> d_seqPyramids;
    Data<SeqUV>	d_seqUVs; ///< List of uv coordinates
    Data<bool> d_computeAllBuffers; ///< Option to call method computeCrossElementBuffers. False by default

protected:
    Size  nbPoints;

    bool validTetrahedra;
    bool validHexahedra;


    /** the array that stores the set of edge-vertex shells, ie for each vertex gives the set of adjacent edges */
    type::vector< EdgesAroundVertex > m_edgesAroundVertex;

    /** the array that stores the set of oriented edge-vertex shells, ie for each vertex gives the set of adjacent edges */
    type::vector< EdgesAroundVertex > m_orientedEdgesAroundVertex;

    /** the array that stores the set of edge-triangle shells, ie for each triangle gives the 3 adjacent edges */
    type::vector< EdgesInTriangle > m_edgesInTriangle;

    /// provides the 4 edges in each quad
    type::vector< EdgesInQuad > m_edgesInQuad;

    /// provides the set of edges for each tetrahedron
    type::vector< EdgesInTetrahedron > m_edgesInTetrahedron;

    /// provides the set of edges for each hexahedron
    type::vector< EdgesInHexahedron > m_edgesInHexahedron;

    /// for each vertex provides the set of triangles adjacent to that vertex
    type::vector< TrianglesAroundVertex > m_trianglesAroundVertex;

    /// for each vertex provides the set of oriented triangles adjacent to that vertex
    type::vector< TrianglesAroundVertex > m_orientedTrianglesAroundVertex;

    /// for each edge provides the set of triangles adjacent to that edge
    type::vector< TrianglesAroundEdge > m_trianglesAroundEdge;

    /// provides the set of triangles adjacent to each tetrahedron
    type::vector< TrianglesInTetrahedron > m_trianglesInTetrahedron;

    /// for each vertex provides the set of quads adjacent to that vertex
    type::vector< QuadsAroundVertex > m_quadsAroundVertex;

    /// for each vertex provides the set of oriented quads adjacent to that vertex
    type::vector< QuadsAroundVertex > m_orientedQuadsAroundVertex;

    /// for each edge provides the set of quads adjacent to that edge
    type::vector< QuadsAroundEdge > m_quadsAroundEdge;

    /// provides the set of quads adjacents to each hexahedron
    type::vector< QuadsInHexahedron > m_quadsInHexahedron;

    /// provides the set of tetrahedrons adjacents to each vertex
    type::vector< TetrahedraAroundVertex> m_tetrahedraAroundVertex;

    /// for each edge provides the set of tetrahedra adjacent to that edge
    type::vector< TetrahedraAroundEdge > m_tetrahedraAroundEdge;

    /// for each triangle provides the set of tetrahedrons adjacent to that triangle
    type::vector< TetrahedraAroundTriangle > m_tetrahedraAroundTriangle;

    /// provides the set of hexahedrons for each vertex
    type::vector< HexahedraAroundVertex > m_hexahedraAroundVertex;

    /// for each edge provides the set of tetrahedra adjacent to that edge
    type::vector< HexahedraAroundEdge > m_hexahedraAroundEdge;

    /// for each quad provides the set of hexahedrons adjacent to that quad
    type::vector< HexahedraAroundQuad > m_hexahedraAroundQuad;

    /** \brief Creates the EdgeSetIndex.
     *
     * This function is only called if the EdgesAroundVertex member is required.
     * m_edgesAroundVertex[i] contains the indices of all edges having the ith DOF as
     * one of their ends.
     */
    void createEdgesAroundVertexArray();

    /** \brief Creates the array of edge indices for each triangle.
     *
     * This function is only called if the EdgesInTriangle array is required.
     * m_edgesInTriangle[i] contains the 3 indices of the 3 edges composing the ith triangle.
     */
    void createEdgesInTriangleArray();

    /** \brief Creates the array of edge indices for each quad.
     *
     * This function is only called if the EdgesInQuad array is required.
     * m_edgesInQuad[i] contains the 4 indices of the 4 edges composing the ith quad.
     */
    void createEdgesInQuadArray();

    /** \brief Creates the array of edge indices for each tetrahedron.
     *
     * This function is only called if the EdgesInTetrahedrone array is required.
     * m_edgesInTetrahedron[i] contains the 6 indices of the 6 edges of each tetrahedron
     The number of each edge is the following : edge 0 links vertex 0 and 1, edge 1 links vertex 0 and 2,
     edge 2 links vertex 0 and 3, edge 3 links vertex 1 and 2, edge 4 links vertex 1 and 3,
     edge 5 links vertex 2 and 3
    */
    void createEdgesInTetrahedronArray();

    /** \brief Creates the array of edge indices for each hexahedron.
     *
     * This function is only called if the EdgesInHexahedron array is required.
     * m_edgesInHexahedron[i] contains the 12 indices of the 12 edges of each hexahedron.
     */
    void createEdgesInHexahedronArray();

    /** \brief Creates the TrianglesAroundVertex Array.
     *
     * This function is only called if the TrianglesAroundVertex array is required.
     * m_trianglesAroundVertex[i] contains the indices of all triangles adjacent to the ith DOF.
     */
    void createTrianglesAroundVertexArray();

    /** \brief Creates the oriented Triangle Vertex Shell Array
    *
    * This function is only called if the OrientedTrianglesAroundVertex array is required.
    * m_orientedTrianglesAroundVertex[i] contains the indices of all triangles adjacent to the ith vertex
    */
    void createOrientedTrianglesAroundVertexArray();

    /** \brief Creates the TrianglesAroundEdge Array.
     *
     * This function is only called if the TrianglesAroundVertex array is required.
     * m_trianglesAroundEdge[i] contains the indices of all triangles adjacent to the ith edge.
     */
    void createTrianglesAroundEdgeArray();

    /** \brief Creates the array of triangle indices for each tetrahedron.
     *
     * This function is only called if the TrianglesInTetrahedron array is required.
     * m_trianglesInTetrahedron[i] contains the 4 indices of the 4 triangles composing the ith tetrahedron.
     */
    void createTrianglesInTetrahedronArray ();

    /** \brief Creates the QuadsAroundVertex Array.
     *
     * This function is only called if the QuadsAroundVertex array is required.
     * m_quadsAroundVertex[i] contains the indices of all quads adjacent to the ith vertex.
     */
    void createQuadsAroundVertexArray ();

    /** \brief Creates the Quad Vertex Shell Array
     *
     * This function is only called if the QuadsAroundVertex array is required.
     * m_quadsAroundVertex[i] contains the indices of all quads adjacent to the ith vertex
     */
    void createOrientedQuadsAroundVertexArray ();

    /** \brief Creates the quadsAroundEdge Array.
     *
     * This function is only called if the QuadsAroundVertex array is required.
     * m_quadsAroundEdge[i] contains the indices of all quads adjacent to the ith edge
     */
    void createQuadsAroundEdgeArray();

    /** \brief Creates the array of quad indices for each hexahedron.
     *
     * This function is only called if the QuadsInHexahedron array is required.
     * m_quadsInHexahedron[i] contains the 6 indices of the 6 quads of each hexahedron.
     */
    void createQuadsInHexahedronArray ();

    /** \brief Creates the TetrahedraAroundVertex Array.
     *
     * This function is only called if the TetrahedraAroundVertex array is required.
     * m_tetrahedraAroundVertex[i] contains the indices of all tetrahedra adjacent to the ith vertex.
     */
    void createTetrahedraAroundVertexArray ();

    /** \brief Creates the TetrahedraAroundEdge Array.
     *
     * This function is only called if the TetrahedraAroundEdge array is required.
     * m_tetrahedraAroundEdge[i] contains the indices of all tetrahedra adjacent to the ith edge.
     */
    void createTetrahedraAroundEdgeArray();

    /** \brief Creates the TetrahedraAroundTriangle Array.
     *
     * This function is only called if the TetrahedraAroundTriangle array is required.
     * m_tetrahedraAroundTriangle[i] contains the indices of all tetrahedra adjacent to the ith triangle.
     */
    void createTetrahedraAroundTriangleArray();

    /** \brief Creates the HexahedraAroundVertex Array.
     *
     * This function is only called if the HexahedraAroundVertex array is required.
     * m_hexahedraAroundVertex[i] contains the indices of all hexahedra adjacent to the ith vertex.
     */
    void createHexahedraAroundVertexArray();

    /** \brief Creates the HexahedraAroundEdge Array.
     *
     * This function is only called if the HexahedraAroundEdge array is required.
     * m_hexahedraAroundEdge[i] contains the indices of all hexahedra adjacent to the ith edge.
     */
    void createHexahedraAroundEdgeArray ();

    /** \brief Creates the HexahedraAroundQuad Array.
     *
     * This function is only called if the HexahedraAroundQuad array is required.
     * m_hexahedraAroundQuad[i] contains the indices of all hexahedra adjacent to the ith quad.
     */
    void createHexahedraAroundQuadArray();

    
public:
    /** \brief Returns the index of the edge joining vertex v1 and vertex v2; returns InvalidID if no edge exists
     *
     */
    EdgeID getEdgeIndex(PointID v1, PointID v2) override;

    /** Returns the indices of a triangle given three vertex indices : returns InvalidID if none */
    TriangleID getTriangleIndex(PointID v1, PointID v2, PointID v3) override;

    /** \brief Returns the index of the quad joining vertex v1, v2, v3 and v4; returns InvalidID if none
     *
     */
    QuadID getQuadIndex(PointID v1, PointID v2, PointID v3,  PointID v4) override;

    /** \brief Returns the index of the tetrahedron given four vertex indices; returns InvalidID if none
     *
     */
    TetrahedronID getTetrahedronIndex(PointID v1, PointID v2, PointID v3, PointID v4) override;

    /** \brief Returns the index of the hexahedron given eight vertex indices; returns InvalidID if none
     *
     */
    HexahedronID getHexahedronIndex(PointID v1, PointID v2, PointID v3, PointID v4, PointID v5, PointID v6, PointID v7, PointID v8) override;

    /** \brief Returns the index (either 0, 1 ,2 or 3) of the vertex whose global index is vertexIndex. Returns -1 if none
    *
    */

    int getVertexIndexInTriangle(const Triangle &t, PointID vertexIndex) const override;

    /** \brief Returns the index (either 0, 1 ,2) of the edge whose global index is edgeIndex. Returns -1 if none
    *
    */
    int getEdgeIndexInTriangle(const EdgesInTriangle &t, EdgeID edgeIndex) const override;

    /** \brief Returns the index (either 0, 1 ,2 or 3) of the vertex whose global index is vertexIndex. Returns -1 if none
    *
    */
    int getVertexIndexInQuad(const Quad &t, PointID vertexIndex) const override;

    /** \brief Returns the index (either 0, 1 ,2, 3) of the edge whose global index is edgeIndex. Returns -1 if none
    *
    */
    int getEdgeIndexInQuad(const EdgesInQuad &t, EdgeID edgeIndex) const override;

    /** \brief Returns the index (either 0, 1 ,2 or 3) of the vertex whose global index is vertexIndex. Returns -1 if none
    *
    */
    int getVertexIndexInTetrahedron(const Tetra &t, PointID vertexIndex) const override;

    /** \brief Returns the index (either 0, 1 ,2 ,3, 4, 5) of the edge whose global index is edgeIndex. Returns -1 if none
    *
    */
    int getEdgeIndexInTetrahedron(const EdgesInTetrahedron &t, EdgeID edgeIndex) const override;

    /** \brief Returns the index (either 0, 1 ,2 ,3) of the triangle whose global index is triangleIndex. Returns -1 if none
    *
    */
    int getTriangleIndexInTetrahedron(const TrianglesInTetrahedron &t, TriangleID triangleIndex) const override;

    /** \brief Returns the index (either 0, 1 ,2, 3, 4, 5, 6, or 7) of the vertex whose global index is vertexIndex. Returns -1 if none
    *
    */
    int getVertexIndexInHexahedron(const Hexa &t, PointID vertexIndex) const override;

    /** \brief Returns the index (either 0, 1 ,2 ,3, 4, 5, 6, 7, 8, 9, 10, 11) of the edge whose global index is edgeIndex. Returns -1 if none
    *
    */
    int getEdgeIndexInHexahedron(const EdgesInHexahedron &t, EdgeID edgeIndex) const override;

    /** \brief Returns the index (either 0, 1 ,2 ,3, 4, 5) of the quad whose global index is quadIndex. Returns -1 if none
    *
    */
    int getQuadIndexInHexahedron(const QuadsInHexahedron &t, QuadID quadIndex) const override;

    /** \brief Returns for each index (between 0 and 5) the two vertex indices that are adjacent to that edge
    *
    */
    Edge getLocalEdgesInTetrahedron (const HexahedronID i) const override;

    /** \brief Returns for each index (between 0 and 12) the two vertex indices that are adjacent to that edge */
    Edge getLocalEdgesInHexahedron (const HexahedronID i) const override;

  	/** \ brief returns the topologyType */
    sofa::geometry::ElementType getTopologyType() const override { return m_upperElementType; }
  
    int revision;

    // To draw the mesh, the topology position must be linked with the mechanical object position 
    Data< bool > d_drawEdges; ///< if true, draw the topology Edges
    Data< bool > d_drawTriangles; ///< if true, draw the topology Triangles
    Data< bool > d_drawQuads; ///< if true, draw the topology Quads
    Data< bool > d_drawTetra; ///< if true, draw the topology Tetrahedra
    Data< bool > d_drawHexa; ///< if true, draw the topology hexahedra

    void invalidate();

    virtual void updateTetrahedra();
    virtual void updateHexahedra();

protected:
    /// Type of higher topology element contains in this container @see ElementType
    sofa::geometry::ElementType m_upperElementType;
};

} //namespace sofa::component::topology::container::constant
