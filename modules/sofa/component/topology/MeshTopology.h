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
#ifndef SOFA_COMPONENT_TOPOLOGY_MESHTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_MESHTOPOLOGY_H

#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <sofa/core/componentmodel/topology/Topology.h>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/vector.h>
#include <sofa/component/component.h>

namespace sofa
{

namespace component
{

namespace container
{
class MeshLoader;
}

namespace topology
{

using namespace sofa::defaulttype;
using helper::vector;
using helper::fixed_array;


class SOFA_COMPONENT_CONTAINER_API MeshTopology : public core::componentmodel::topology::BaseMeshTopology
{
public:

    MeshTopology();

    virtual void parse(core::objectmodel::BaseObjectDescription* arg);

    virtual void init();

    virtual int getNbPoints() const;

    virtual void setNbPoints(int n);

    // Complete sequence accessors

    virtual const SeqEdges& getEdges();
    virtual const SeqTriangles& getTriangles();
    virtual const SeqQuads& getQuads();
    virtual const SeqTetrahedra& getTetrahedra();
    virtual const SeqHexahedra& getHexahedra();

    // Random accessors

    virtual int getNbEdges();
    virtual int getNbTriangles();
    virtual int getNbQuads();
    virtual int getNbTetrahedra();
    virtual int getNbHexahedra();

    virtual const Edge getEdge(EdgeID i);
    virtual const Triangle getTriangle(TriangleID i);
    virtual const Quad getQuad(QuadID i);
    virtual const Tetra getTetrahedron(TetraID i);
    virtual const Hexa getHexahedron(HexaID i);

    /// @name neighbors queries
    /// @{
    /// Returns the set of edges adjacent to a given vertex.
    virtual const EdgesAroundVertex &getEdgesAroundVertex(PointID i);
    /// Returns the set of edges adjacent to a given vertex.
    virtual const EdgesAroundVertex &getOrientedEdgesAroundVertex(PointID i);
    /// Returns the set of edges adjacent to a given triangle.
    virtual const EdgesInTriangle &getEdgesInTriangle(TriangleID i);
    /// Returns the set of edges adjacent to a given quad.
    virtual const EdgesInQuad &getEdgesInQuad(QuadID i);
    /// Returns the set of edges adjacent to a given tetrahedron.
    virtual const EdgesInTetrahedron& getEdgesInTetrahedron(TetraID i);
    /// Returns the set of edges adjacent to a given hexahedron.
    virtual const EdgesInHexahedron& getEdgesInHexahedron(HexaID i);
    /// Returns the set of triangle adjacent to a given vertex.
    virtual const TrianglesAroundVertex &getTrianglesAroundVertex(PointID i);
    /// Returns the set of oriented triangle adjacent to a given vertex.
    virtual const TrianglesAroundVertex &getOrientedTrianglesAroundVertex(PointID i);
    /// Returns the set of triangle adjacent to a given edge.
    virtual const TrianglesAroundEdge &getTrianglesAroundEdge(EdgeID i);
    /// Returns the set of triangles adjacent to a given tetrahedron.
    virtual const TrianglesInTetrahedron& getTrianglesInTetrahedron(TetraID i);
    /// Returns the set of quad adjacent to a given vertex.
    virtual const QuadsAroundVertex &getQuadsAroundVertex(PointID i);
    /// Returns the set of oriented quad adjacent to a given vertex.
    virtual const QuadsAroundVertex &getOrientedQuadsAroundVertex(PointID i);
    /// Returns the set of quad adjacent to a given edge.
    virtual const QuadsAroundEdge &getQuadsAroundEdge(EdgeID i);
    /// Returns the set of quads adjacent to a given hexahedron.
    virtual const QuadsInHexahedron& getQuadsInHexahedron(HexaID i);
    /// Returns the set of tetrahedra adjacent to a given vertex.
    virtual const TetrahedraAroundVertex& getTetrahedraAroundVertex(PointID i);
    /// Returns the set of tetrahedra adjacent to a given edge.
    virtual const TetrahedraAroundEdge& getTetrahedraAroundEdge(EdgeID i);
    /// Returns the set of tetrahedra adjacent to a given triangle.
    virtual const TetrahedraAroundTriangle& getTetrahedraAroundTriangle(TriangleID i);
    /// Returns the set of hexahedra adjacent to a given vertex.
    virtual const HexahedraAroundVertex& getHexahedraAroundVertex(PointID i);
    /// Returns the set of hexahedra adjacent to a given edge.
    virtual const HexahedraAroundEdge& getHexahedraAroundEdge(EdgeID i);
    /// Returns the set of hexahedra adjacent to a given quad.
    virtual const HexahedraAroundQuad& getHexahedraAroundQuad(QuadID i);
    /// @}

    // Points accessors (not always available)

    virtual bool hasPos() const;
    virtual double getPX(int i) const;
    virtual double getPY(int i) const;
    virtual double getPZ(int i) const;

    // for procedural creation without file loader
    virtual void clear();
    void addPoint(double px, double py, double pz);
    void addEdge( int a, int b );
    void addTriangle( int a, int b, int c );
    void addQuad( int a, int b, int c, int d );
    void addTetra( int a, int b, int c, int d );
    void addHexa( int a, int b, int c, int d, int e, int f, int g, int h );

    /// get the current revision of this mesh (use to detect changes)
    int getRevision() const { return revision; }

    /// return true if the given cube is active, i.e. it contains or is surrounded by mapped points.
    /// @deprecated
    virtual bool isCubeActive(int /*index*/) { return true; }

    void draw();

    virtual bool hasVolume() { return ( ( getNbTetrahedra() + getNbHexahedra() ) > 0 ); }
    virtual bool hasSurface() { return ( ( getNbTriangles() + getNbQuads() ) > 0 ); }
    virtual bool hasLines() { return ( ( getNbLines() ) > 0 ); }

    virtual bool isVolume() { return hasVolume(); }
    virtual bool isSurface() { return !hasVolume() && hasSurface(); }
    virtual bool isLines() { return !hasVolume() && !hasSurface() && hasLines(); }

    // test whether p0p1 has the same orientation as triangle t
    // opposite dirction: return -1
    // same direction: return 1
    // otherwise: return 0
    int computeRelativeOrientationInTri(const unsigned int ind_p0, const unsigned int ind_p1, const unsigned int ind_t);

    // test whether p0p1 has the same orientation as triangle t
    // opposite dirction: return -1
    // same direction: return 1
    // otherwise: return 0
    int computeRelativeOrientationInQuad(const unsigned int ind_p0, const unsigned int ind_p1, const unsigned int ind_q);

protected:
    int nbPoints;
    Data< vector< defaulttype::Vec<3,SReal> > > seqPoints;

    Data<SeqEdges> seqEdges;
    bool validEdges;

    //SeqTriangles   seqTriangles;
    Data<SeqTriangles> seqTriangles;
    bool         validTriangles;

    //SeqQuads       seqQuads;
    Data<SeqQuads>       seqQuads;
    bool         validQuads;

    //SeqTetrahedra      seqTetras;
    Data<SeqTetrahedra>      seqTetras;
    bool         validTetras;

#ifdef SOFA_NEW_HEXA
    //SeqHexahedra	   seqHexas;
    Data<SeqHexahedra>	   seqHexas;
#else
    Data<SeqCubes>       seqHexas;
#endif
    bool         validHexas;

    /** the array that stores the set of edge-vertex shells, ie for each vertex gives the set of adjacent edges */
    vector< EdgesAroundVertex > m_edgesAroundVertex;

    /** the array that stores the set of oriented edge-vertex shells, ie for each vertex gives the set of adjacent edges */
    vector< EdgesAroundVertex > m_orientedEdgesAroundVertex;

    /** the array that stores the set of edge-triangle shells, ie for each triangle gives the 3 adjacent edges */
    vector< EdgesInTriangle > m_edgeTriangleShell;

    /// provides the 4 edges in each quad
    vector< EdgesInQuad > m_edgeQuadShell;

    /// provides the set of edges for each tetrahedron
    vector< EdgesInTetrahedron > m_edgeTetraShell;

    /// provides the set of edges for each hexahedron
    vector< EdgesInHexahedron > m_edgeHexaShell;

    /// for each vertex provides the set of triangles adjacent to that vertex
    vector< TrianglesAroundVertex > m_trianglesAroundVertex;

    /// for each vertex provides the set of oriented triangles adjacent to that vertex
    vector< TrianglesAroundVertex > m_orientedTrianglesAroundVertex;

    /// for each edge provides the set of triangles adjacent to that edge
    vector< TrianglesAroundEdge > m_trianglesAroundEdge;

    /// provides the set of triangles adjacent to each tetrahedron
    vector< TrianglesInTetrahedron > m_triangleTetraShell;

    /// for each vertex provides the set of quads adjacent to that vertex
    vector< QuadsAroundVertex > m_quadsAroundVertex;

    /// for each vertex provides the set of oriented quads adjacent to that vertex
    vector< QuadsAroundVertex > m_orientedQuadsAroundVertex;

    /// for each edge provides the set of quads adjacent to that edge
    vector< QuadsAroundEdge > m_quadsAroundEdge;

    /// provides the set of quads adjacents to each hexahedron
    vector< QuadsInHexahedron > m_quadHexaShell;

    /// provides the set of tetrahedrons adjacents to each vertex
    vector< TetrahedraAroundVertex> m_tetrahedraAroundVertex;

    /// for each edge provides the set of tetras adjacent to that edge
    vector< TetrahedraAroundEdge > m_tetrahedraAroundEdge;

    /// for each triangle provides the set of tetrahedrons adjacent to that triangle
    vector< TetrahedraAroundTriangle > m_tetrahedraAroundTriangle;

    /// provides the set of hexahedrons for each vertex
    vector< HexahedraAroundVertex > m_hexahedraAroundVertex;

    /// for each edge provides the set of tetras adjacent to that edge
    vector< HexahedraAroundEdge > m_hexahedraAroundEdge;

    /// for each quad provides the set of hexahedrons adjacent to that quad
    vector< HexahedraAroundQuad > m_hexahedraAroundQuad;

    /** \brief Creates the EdgeSetIndex.
     *
     * This function is only called if the EdgesAroundVertex member is required.
     * m_edgesAroundVertex[i] contains the indices of all edges having the ith DOF as
     * one of their ends.
     */
    void createEdgesAroundVertexArray();

    /** \brief Creates the array of edge indices for each triangle
     *
     * This function is only called if the EdgeTriangleShell array is required.
     * m_edgeTriangleShell[i] contains the 3 indices of the 3 edges opposite to the ith triangle
     */
    void createEdgeTriangleShellArray();

    /** \brief Creates the array of edge indices for each quad
     *
     * This function is only called if the EdgeQuadShell array is required.
     * m_edgeQuadShell[i] contains the 4 indices of the 4 edges opposite to the ith Quad
     */
    void createEdgeQuadShellArray();

    /** \brief Creates the array of edge indices for each tetrahedron
     *
     * This function is only called if the EdgeTetraShell array is required.
     * m_edgeTetraShell[i] contains the indices of the edges to the ith tetrahedron
     */
    void createEdgeTetraShellArray();

    /** \brief Creates the array of edge indices for each hexahedrom
     *
     * This function is only called if the EdgeHexaShell array is required.
     * m_edgeHexaShell[i] contains the indices of the edges to the ith hexahedrom
     */
    void createEdgeHexaShellArray();

    /** \brief Creates the Triangle Vertex Shell Array
     *
     * This function is only called if the TrianglesAroundVertex array is required.
     * m_trianglesAroundVertex[i] contains the indices of all triangles adjacent to the ith vertex
     */
    void createTrianglesAroundVertexArray();

    /** \brief Creates the oriented Triangle Vertex Shell Array
    *
    * This function is only called if the OrientedTrianglesAroundVertex array is required.
    * m_orientedTrianglesAroundVertex[i] contains the indices of all triangles adjacent to the ith vertex
    */
    void createOrientedTrianglesAroundVertexArray();

    /** \brief Creates the Triangle Edge Shell Array
     *
     * This function is only called if the TrianglesAroundEdge array is required.
     * m_trianglesAroundEdge[i] contains the indices of all triangles adjacent to the ith edge
     */
    void createTrianglesAroundEdgeArray();

    /** \brief Creates the array of triangle indices for each tetrahedron
     *
     * This function is only called if the TriangleTetraShell array is required.
     * m_triangleTetraShell[i] contains the indices of the triangles to the ith tetrahedron
     */
    void createTriangleTetraShellArray ();

    /** \brief Creates the Quad Vertex Shell Array
     *
     * This function is only called if the QuadsAroundVertex array is required.
     * m_quadsAroundVertex[i] contains the indices of all quads adjacent to the ith vertex
     */
    void createQuadsAroundVertexArray ();

    /** \brief Creates the Quad Vertex Shell Array
     *
     * This function is only called if the QuadsAroundVertex array is required.
     * m_quadsAroundVertex[i] contains the indices of all quads adjacent to the ith vertex
     */
    void createOrientedQuadsAroundVertexArray ();

    /** \brief Creates the Quad Edge Shell Array
     *
     * This function is only called if the QuadsAroundEdge array is required.
     * m_quadsAroundEdge[i] contains the indices of all quads adjacent to the ith edge
     */
    void createQuadsAroundEdgeArray();

    /** \brief Creates the array of quad indices for each hexahedrom
     *
     * This function is only called if the QuadHexaShell array is required.
     * m_quadHexaShell[i] contains the indices of the quads to the ith hexahedrom
     */
    void createQuadHexaShellArray ();

    /** \brief Creates the array of tetrahedron indices for each vertex
     *
     * This function is only called if the TetrahedraAroundVertex array is required.
     * m_tetrahedraAroundVertex[i] contains the indices of the tetras to the ith vertex
     */
    void createTetrahedraAroundVertexArray ();

    /** \brief Creates the array of tetrahedron indices for each edge
     *
     * This function is only called if the TetrahedraAroundEdge array is required.
     * m_tetrahedraAroundEdge[i] contains the indices of the tetrahedrons to the ith edge
     */
    void createTetrahedraAroundEdgeArray();

    /** \brief Creates the array of tetrahedron indices adjacent to each triangle
     *
     * This function is only called if the TetrahedraAroundTriangle array is required.
     * m_tetrahedraAroundTriangle[i] contains the indices of the tetrahedrons adjacent to the ith triangle
     */
    void createTetrahedraAroundTriangleArray();

    /** \brief Creates the array of hexahedron indices for each vertex
     *
     * This function is only called if the HexahedraAroundVertex array is required.
     * m_hexahedraAroundVertex[i] contains the indices of the hexas to the ith vertex
     */
    void createHexahedraAroundVertexArray();

    /** \brief Creates the array of hexahedron indices for each edge
     *
     * This function is only called if the HexahedraAroundEdge array is required.
     * m_hexahedraAroundEdge[i] contains the indices of the hexahedrons to the ith edge
     */
    void createHexahedraAroundEdgeArray ();

    /** \brief Creates the array of hexahedron indices adjacent to each quad
     *
     * This function is only called if the HexahedraAroundQuad array is required.
     * m_hexahedraAroundQuad[i] contains the indices of the hexahedrons adjacent to the ith quad
     */
    void createHexahedraAroundQuadArray();



    /** \brief Returns the Triangle Vertex Shells array.
     *
     */
    const vector< TrianglesAroundVertex > &getTrianglesAroundVertexArray() ;

    /** \brief Returns the Quad Vertex Shells array.
     *
     */
    const vector< QuadsAroundVertex >& getQuadsAroundVertexArray();


public:
    /** \brief Returns the index of the edge joining vertex v1 and vertex v2; returns -1 if no edge exists
     *
     */
    int getEdgeIndex(PointID v1, PointID v2);

protected:
    /** Returns the indices of a triangle given three vertex indices : returns -1 if none */
    int getTriangleIndex(PointID v1, PointID v2, PointID v3);

    /** \brief Returns the index of the quad joining vertex v1, v2, v3 and v4; returns -1 if no edge exists
     *
     */
    int getQuadIndex(PointID v1, PointID v2, PointID v3,  PointID v4);

    /** \brief Returns the index of the tetrahedron given four vertex indices; returns -1 if no edge exists
     *
     */
    int getTetrahedronIndex(PointID v1, PointID v2, PointID v3, PointID v4);

    /** \brief Returns the index of the hexahedron given eight vertex indices; returns -1 if no edge exists
     *
     */
    int getHexahedronIndex(PointID v1, PointID v2, PointID v3, PointID v4, PointID v5, PointID v6, PointID v7, PointID v8);

    /** \brief Returns the index (either 0, 1 ,2 or 3) of the vertex whose global index is vertexIndex. Returns -1 if none
    *
    */

    int getVertexIndexInTriangle(const Triangle &t, PointID vertexIndex) const;

    /** \brief Returns the index (either 0, 1 ,2) of the edge whose global index is edgeIndex. Returns -1 if none
    *
    */
    int getEdgeIndexInTriangle(const EdgesInTriangle &t, EdgeID edgeIndex) const;

    /** \brief Returns the index (either 0, 1 ,2 or 3) of the vertex whose global index is vertexIndex. Returns -1 if none
    *
    */
    int getVertexIndexInQuad(Quad &t, PointID vertexIndex) const;

    /** \brief Returns the index (either 0, 1 ,2, 3) of the edge whose global index is edgeIndex. Returns -1 if none
    *
    */
    int getEdgeIndexInQuad(EdgesInQuad &t, EdgeID edgeIndex) const;

    /** \brief Returns the index (either 0, 1 ,2 or 3) of the vertex whose global index is vertexIndex. Returns -1 if none
    *
    */
    int getVertexIndexInTetrahedron(const Tetra &t, PointID vertexIndex) const;

    /** \brief Returns the index (either 0, 1 ,2 ,3, 4, 5) of the edge whose global index is edgeIndex. Returns -1 if none
    *
    */
    int getEdgeIndexInTetrahedron(const EdgesInTetrahedron &t, EdgeID edgeIndex) const;

    /** \brief Returns the index (either 0, 1 ,2 ,3) of the triangle whose global index is triangleIndex. Returns -1 if none
    *
    */
    int getTriangleIndexInTetrahedron(const TrianglesInTetrahedron &t, TriangleID triangleIndex) const;

    /** \brief Returns the index (either 0, 1 ,2, 3, 4, 5, 6, or 7) of the vertex whose global index is vertexIndex. Returns -1 if none
    *
    */
    int getVertexIndexInHexahedron(Hexa &t, PointID vertexIndex) const;

    /** \brief Returns the index (either 0, 1 ,2 ,3, 4, 5, 6, 7, 8, 9, 10, 11) of the edge whose global index is edgeIndex. Returns -1 if none
    *
    */
    int getEdgeIndexInHexahedron(const EdgesInHexahedron &t, EdgeID edgeIndex) const;

    /** \brief Returns the index (either 0, 1 ,2 ,3, 4, 5) of the quad whose global index is quadIndex. Returns -1 if none
    *
    */
    int getQuadIndexInHexahedron(const QuadsInHexahedron &t, QuadID quadIndex) const;

    /** \brief Returns for each index (between 0 and 5) the two vertex indices that are adjacent to that edge
    *
    */
    Edge getLocalEdgesInTetrahedron (const unsigned int i) const;

    int revision;

    Data< bool > _draw;

    void invalidate();

    virtual void updateEdges();
    virtual void updateTriangles();
    virtual void updateQuads();
    virtual void updateTetras();
    virtual void updateHexas();

protected:
    virtual void loadFromMeshLoader(sofa::component::container::MeshLoader* loader);
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
