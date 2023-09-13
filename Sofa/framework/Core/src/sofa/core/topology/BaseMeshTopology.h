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

#include <sofa/core/fwd.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/core/objectmodel/DataFileName.h>

namespace sofa::core::topology
{

class SOFA_CORE_API BaseMeshTopology : public core::topology::Topology
{
public:
    SOFA_ABSTRACT_CLASS(BaseMeshTopology, core::topology::Topology);
    SOFA_BASE_CAST_IMPLEMENTATION(BaseMeshTopology)

    typedef sofa::type::vector<Edge> 		        SeqEdges;
    typedef sofa::type::vector<Triangle>		    SeqTriangles;
    typedef sofa::type::vector<Quad>		        SeqQuads;
    typedef sofa::type::vector<Tetra>		        SeqTetrahedra;
    typedef sofa::type::vector<Hexa>		        SeqHexahedra;

    /// @name Deprecated types, for backward-compatibility
    /// @{
    typedef EdgeID		                LineID;
    typedef Edge		                Line;
    typedef SeqEdges	                SeqLines;
    /// @}

    /// fixed-size neighbors arrays
    /// @{
    typedef sofa::type::fixed_array<EdgeID,3>		EdgesInTriangle;
    typedef sofa::type::fixed_array<EdgeID,4>		EdgesInQuad;
    typedef sofa::type::fixed_array<TriangleID,4>	TrianglesInTetrahedron;
    typedef sofa::type::fixed_array<EdgeID,6>		EdgesInTetrahedron;
    typedef sofa::type::fixed_array<QuadID,6>		QuadsInHexahedron;
    typedef sofa::type::fixed_array<EdgeID,12>    EdgesInHexahedron;

    static constexpr EdgesInTriangle        InvalidEdgesInTriangles       = type::makeHomogeneousArray<EdgesInTriangle>(sofa::InvalidID);
    static constexpr EdgesInQuad            InvalidEdgesInQuad            = type::makeHomogeneousArray<EdgesInQuad>(sofa::InvalidID);
    static constexpr TrianglesInTetrahedron InvalidTrianglesInTetrahedron = type::makeHomogeneousArray<TrianglesInTetrahedron>(sofa::InvalidID);
    static constexpr EdgesInTetrahedron     InvalidEdgesInTetrahedron     = type::makeHomogeneousArray<EdgesInTetrahedron>(sofa::InvalidID);
    static constexpr QuadsInHexahedron      InvalidQuadsInHexahedron      = type::makeHomogeneousArray<QuadsInHexahedron>(sofa::InvalidID);
    static constexpr EdgesInHexahedron      InvalidEdgesInHexahedron      = type::makeHomogeneousArray<EdgesInHexahedron>(sofa::InvalidID);

    /// @}

    /// dynamic-size neighbors arrays
    /// @{
    typedef sofa::type::vector<PointID>		    VerticesAroundVertex;
    typedef sofa::type::vector<EdgeID>			EdgesAroundVertex;
    typedef sofa::type::vector<TriangleID>	    TrianglesAroundVertex;
    typedef sofa::type::vector<QuadID>			QuadsAroundVertex;
    typedef sofa::type::vector<TetraID>		    TetrahedraAroundVertex;
    typedef sofa::type::vector<HexaID>			HexahedraAroundVertex;
    typedef sofa::type::vector<TriangleID>	    TrianglesAroundEdge;
    typedef sofa::type::vector<QuadID>			QuadsAroundEdge;
    typedef sofa::type::vector<TetraID>		    TetrahedraAroundEdge;
    typedef sofa::type::vector<HexaID>			HexahedraAroundEdge;
    typedef sofa::type::vector<TetraID>		    TetrahedraAroundTriangle;
    typedef sofa::type::vector<HexaID>			HexahedraAroundQuad;
    /// @}
protected:
    BaseMeshTopology()	;
public:
    void init() override;

    /// Load the topology from a file.
    ///
    /// The default implementation supports the following formats: obj, gmsh, mesh (custom simple text file), xs3 (deprecated description of mass-springs networks).
    virtual bool load(const char* filename);
    virtual std::string getFilename() const {return fileTopology.getValue();}

    // defined in Topology
    //virtual int getNbPoints() const = 0;

    /// Complete sequence accessors
    /// @{
    virtual const SeqEdges& getEdges() = 0;
    virtual const SeqTriangles& getTriangles() = 0;
    virtual const SeqQuads& getQuads() = 0;
    virtual const SeqTetrahedra& getTetrahedra() = 0;
    virtual const SeqHexahedra& getHexahedra() = 0;
    /// @}

    /// Random accessors
    /// @{

    virtual Size getNbEdges()                   { return Size(getEdges().size()); }
    virtual Size getNbTriangles()               { return Size(getTriangles().size()); }
    virtual Size getNbQuads()                   { return Size(getQuads().size()); }
    virtual Size getNbTetrahedra()              { return Size(getTetrahedra().size()); }
    virtual Size getNbHexahedra()	              { return Size(getHexahedra().size()); }

    virtual const Edge getEdge(EdgeID i)             { return getEdges()[i]; }
    virtual const Triangle getTriangle(TriangleID i) { return getTriangles()[i]; }
    virtual const Quad getQuad(QuadID i)             { return getQuads()[i]; }
    virtual const Tetra getTetrahedron(TetraID i)    { return getTetrahedra()[i]; }
    virtual const Hexa getHexahedron(HexaID i)       { return getHexahedra()[i]; }   
	   
    /// Type of higher topology element contains in this container @see ElementType
    virtual sofa::geometry::ElementType getTopologyType() const = 0;
    /// @}

    /// Bridge from old functions (using Tetra/Tetras and Hexa/Hexas) to new ones
    ///@{
    virtual Size getNbTetras()    { return getNbTetrahedra(); }
    virtual Size getNbHexas()     { return getNbHexahedra(); }

    virtual Tetra getTetra(TetraID i)          { return getTetrahedra()[i]; }
    virtual Hexa getHexa(HexaID i)             { return getHexahedra()[i]; }

    virtual const SeqTetrahedra& getTetras() {return getTetrahedra();}
    virtual const SeqHexahedra& getHexas() {return getHexahedra();}
    /// @}

    /// @name neighbors queries
    /// @{
    /// Returns the set of edges adjacent to a given vertex.
    virtual const EdgesAroundVertex& getEdgesAroundVertex(PointID i);
    /// Returns the set of edges adjacent to a given triangle.
    virtual const EdgesInTriangle& getEdgesInTriangle(TriangleID i);
    /// Returns the set of edges adjacent to a given quad.
    virtual const EdgesInQuad& getEdgesInQuad(QuadID i);
    /// Returns the set of edges adjacent to a given tetrahedron.
    virtual const EdgesInTetrahedron& getEdgesInTetrahedron(TetraID i);
    /// Returns the set of edges adjacent to a given hexahedron.
    virtual const EdgesInHexahedron& getEdgesInHexahedron(HexaID i);
    /// Returns the set of triangles adjacent to a given vertex.
    virtual const TrianglesAroundVertex& getTrianglesAroundVertex(PointID i);
    /// Returns the set of triangles adjacent to a given edge.
    virtual const TrianglesAroundEdge& getTrianglesAroundEdge(EdgeID i);
    /// Returns the set of triangles adjacent to a given tetrahedron.
    virtual const TrianglesInTetrahedron& getTrianglesInTetrahedron(TetraID i);
    /// Returns the set of quads adjacent to a given vertex.
    virtual const QuadsAroundVertex& getQuadsAroundVertex(PointID i);
    /// Returns the set of quads adjacent to a given edge.
    virtual const QuadsAroundEdge& getQuadsAroundEdge(EdgeID i);
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

    /// Returns the set of vertices adjacent to a given vertex (i.e. sharing an edge)
    virtual const VerticesAroundVertex getVerticesAroundVertex(PointID i);
    /// Returns the set of element indices adjacent to a given element (i.e. sharing a link)
    virtual const sofa::type::vector<Index> getElementAroundElement(Index elem);
    /// Returns the set of element indices adjacent to a given list of elements (i.e. sharing a link)
    virtual const sofa::type::vector<Index> getElementAroundElements(sofa::type::vector<Index> elems);
    /// @}


    /// Returns the index of the edge joining vertex v1 and vertex v2; returns InvalidID if no edge exists
    virtual EdgeID getEdgeIndex(PointID v1, PointID v2);
    /// Returns the index of the triangle given three vertex indices; returns InvalidID if no triangle exists
    virtual TriangleID getTriangleIndex(PointID v1, PointID v2, PointID v3);
    /// Returns the index of the quad given four vertex indices; returns InvalidID if no quad exists
    virtual QuadID getQuadIndex(PointID v1, PointID v2, PointID v3, PointID v4);
    /// Returns the index of the tetrahedron given four vertex indices; returns InvalidID if no tetrahedron exists
    virtual TetrahedronID getTetrahedronIndex(PointID v1, PointID v2, PointID v3, PointID v4);
    /// Returns the index of the hexahedron given eight vertex indices; returns InvalidID if no hexahedron exists
    virtual HexahedronID getHexahedronIndex(PointID v1, PointID v2, PointID v3, PointID v4, PointID v5, PointID v6, PointID v7, PointID v8);


    /** returns the index (either 0, 1 ,2 or 3) of the vertex whose global index is vertexIndex. Returns -1 if none */
    virtual int getVertexIndexInTriangle(const Triangle &t, PointID vertexIndex) const;
    /** returns the index (either 0, 1 ,2) of the edge whose global index is edgeIndex. Returns -1 if none */
    virtual int getEdgeIndexInTriangle(const EdgesInTriangle &t, EdgeID edgeIndex) const;

    /** returns the index (either 0, 1 ,2 or 3) of the vertex whose global index is vertexIndex. Returns -1 if none */
    virtual int getVertexIndexInQuad(const Quad &t, PointID vertexIndex) const;
    /** returns the index (either 0, 1 ,2, 3) of the edge whose global index is edgeIndex. Returns -1 if none */
    virtual int getEdgeIndexInQuad(const EdgesInQuad &t, EdgeID edgeIndex) const;

    /** returns the index (either 0, 1 ,2 or 3) of the vertex whose global index is vertexIndex. Returns -1 if none */
    virtual int getVertexIndexInTetrahedron(const Tetra &t, PointID vertexIndex) const;
    /** returns the index (either 0, 1 ,2 ,3, 4, 5) of the edge whose global index is edgeIndex. Returns -1 if none */
    virtual int getEdgeIndexInTetrahedron(const EdgesInTetrahedron &t, EdgeID edgeIndex) const;
    /** returns the index (either 0, 1 ,2 ,3) of the triangle whose global index is triangleIndex. Returns -1 if none */
    virtual int getTriangleIndexInTetrahedron(const TrianglesInTetrahedron &t, TriangleID triangleIndex) const;

    /** returns the index (either 0, 1 ,2, 3, 4, 5, 6, or 7) of the vertex whose global index is vertexIndex. Returns -1 if none */
    virtual int getVertexIndexInHexahedron(const Hexa &t, PointID vertexIndex) const;
    /** returns the index (either 0, 1 ,2 ,3, 4, 5, 6, 7, 8, 9, 10, 11) of the edge whose global index is edgeIndex. Returns -1 if none */
    virtual int getEdgeIndexInHexahedron(const EdgesInHexahedron &t, EdgeID edgeIndex) const;
    /** returns the index (either 0, 1 ,2 ,3, 4, 5) of the quad whose global index is quadIndex. Returns -1 if none */
    virtual int getQuadIndexInHexahedron(const QuadsInHexahedron &t, QuadID quadIndex) const;

    /** \brief Returns for each index (between 0 and 5) the two vertex indices that are adjacent to that edge */
    virtual Edge getLocalEdgesInTetrahedron (const PointID i) const;
    /** \brief Returns for each index (between 0 and 3) the three local vertices indices that are adjacent to that triangle */
    virtual Triangle getLocalTrianglesInTetrahedron (const PointID i) const;

    /** \brief Returns for each index (between 0 and 12) the two vertex indices that are adjacent to that edge */
    virtual Edge getLocalEdgesInHexahedron (const PointID i) const;
    /** \brief Returns for each index (between 0 and 6) the four vertices indices that are adjacent to that quad */
    virtual Quad getLocalQuadsInHexahedron (const PointID i) const;

    /// @name Deprecated names, for backward-compatibility
    /// @{
    const SeqLines& getLines() { return getEdges(); }
    Size getNbLines() { return getNbEdges(); }
    Line getLine(LineID i) { return getEdge(i); }
    /// @}

    /// @name Initial points accessors (only available if the topology was loaded from a file containing this information).
    /// Note that this data is only used for initialization and is not maintained afterwards (i.e. topological changes may not be applied)
    /// @{
    bool hasPos() const override { return false; }
    SReal getPX(Index) const override { return 0.0; }
    SReal getPY(Index) const override { return 0.0; }
    SReal getPZ(Index) const override { return 0.0; }
    /// @}

    /// Procedural creation methods
    /// @{
    virtual void clear();
    virtual void addPoint(SReal px, SReal py, SReal pz);
    virtual void addEdge( Index a, Index b );
    void addLine( Index a, Index b ) { addEdge(a,b); }
    virtual void addTriangle( Index a, Index b, Index c );
    virtual void addQuad( Index a, Index b, Index c, Index d );
    virtual void addTetra( Index a, Index b, Index c, Index d );
    virtual void addHexa( Index a, Index b, Index c, Index d, Index e, Index f, Index g, Index h );
    /// @}

    /// get information about connexity of the mesh
    /// @{
    /// Checks if the topology has only one connected component. @return Return true if so.
    virtual bool checkConnexity() { return true; }

    /// Checks if the topology is coherent. @return true if so. Should be override by child class.
    virtual bool checkTopology() const { return true; }

    /// Returns the number of connected component.
    virtual Size getNumberOfConnectedComponent() {return 0;}
    /// Returns the set of element indices connected to an input one (i.e. which can be reached by topological links)
    virtual const sofa::type::vector<Index> getConnectedElement(Index elem);
    /// @}

    /// Get the current revision of this mesh.
    /// This can be used to detect changes, however topological changes event should be used whenever possible.
    virtual int getRevision() const { return 0; }

    /// Will change order of vertices in triangle: t[1] <=> t[2]
    virtual void reOrientateTriangle(TriangleID id);

    /// Management of topological changes and state changes
    /// @{

    /** \brief Provides an iterator on the first element in the list of TopologyChange objects.
    */
    virtual std::list<const TopologyChange *>::const_iterator beginChange() const;

    /** \brief Provides an iterator on the last element in the list of TopologyChange objects.
    */
    virtual std::list<const TopologyChange *>::const_iterator endChange() const;

    /** \brief Provides an iterator on the first element in the list of StateChange objects.
    */
    virtual std::list<const TopologyChange *>::const_iterator beginStateChange() const;

    /** \brief Provides an iterator on the last element in the list of StateChange objects.
    */
    virtual std::list<const TopologyChange *>::const_iterator endStateChange() const;

    /// @}

    // functions returning border elements. To be moved in a mapping.
    virtual const sofa::type::vector<TriangleID>& getTrianglesOnBorder();

    virtual const sofa::type::vector<EdgeID>& getEdgesOnBorder();

    virtual const sofa::type::vector<PointID>& getPointsOnBorder();

protected:

    sofa::core::objectmodel::DataFileName fileTopology;

public:

    bool insertInNode( objectmodel::BaseNode* node ) override;
    bool removeInNode( objectmodel::BaseNode* node ) override;

};
} // namespace sofa::core::topology
