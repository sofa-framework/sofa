/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_CORE_COMPONENTMODEL_TOPOLOGY_BASEMESHTOPOLOGY_H
#define SOFA_CORE_COMPONENTMODEL_TOPOLOGY_BASEMESHTOPOLOGY_H

#include <stdlib.h>
#include <vector>
#include <list>
#include <string>
#include <iostream>
#include <sofa/core/componentmodel/topology/Topology.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/vector.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace topology
{

using namespace sofa::defaulttype;
using helper::vector;
using helper::fixed_array;

//#define SOFA_NEW_HEXA

class BaseMeshTopology : public core::componentmodel::topology::Topology
{
public:
    //typedef int index_type;
    typedef unsigned int index_type;
    enum { InvalidID = (unsigned)-1 };
    typedef index_type PointID;
    typedef index_type EdgeID;
    typedef index_type TriangleID;
    typedef index_type QuadID;
    typedef index_type TetraID;
    typedef index_type HexaID;

    typedef fixed_array<PointID,2> Edge;
    typedef fixed_array<PointID,3> Triangle;
    typedef fixed_array<PointID,4> Quad;
    typedef fixed_array<PointID,4> Tetra;
    typedef fixed_array<PointID,8> Hexa;

    typedef vector<Edge> SeqEdges;
    typedef vector<Triangle> SeqTriangles;
    typedef vector<Quad> SeqQuads;
    typedef vector<Tetra> SeqTetras;
    typedef vector<Hexa> SeqHexas;

    /// @name Deprecated types, for backward-compatibility
    /// @{
    typedef EdgeID LineID;
    typedef Edge Line;
    typedef SeqEdges SeqLines;
#ifndef SOFA_NEW_HEXA
    typedef HexaID CubeID;
    typedef Hexa Cube;
    typedef SeqHexas SeqCubes;
#endif
    /// @}

    /// fixed-size neighbors arrays
    /// @{
    typedef fixed_array<EdgeID,3> TriangleEdges;
    typedef fixed_array<QuadID,4> QuadEdges;
    typedef fixed_array<TriangleID,4> TetraTriangles;
    typedef fixed_array<EdgeID,6> TetraEdges;
    typedef fixed_array<QuadID,6> HexaQuads;
    typedef fixed_array<EdgeID,12> HexaEdges;
    /// @}

    /// dynamic-size neighbors arrays
    /// @{
    typedef vector<EdgeID> VertexEdges;
    typedef vector<TriangleID> VertexTriangles;
    typedef vector<QuadID> VertexQuads;
    typedef vector<TetraID> VertexTetras;
    typedef vector<HexaID> VertexHexas;
    typedef vector<TriangleID> EdgeTriangles;
    typedef vector<QuadID> EdgeQuads;
    typedef vector<TetraID> EdgeTetras;
    typedef vector<HexaID> EdgeHexas;
    typedef vector<TetraID> TriangleTetras;
    typedef vector<HexaID> QuadHexas;
    /// @}

    BaseMeshTopology();

    virtual bool load(const char* filename) = 0;

    // defined in Topology
    //virtual int getNbPoints() const = 0;

    /// Complete sequence accessors
    /// @{
    virtual const SeqEdges& getEdges() = 0;
    virtual const SeqTriangles& getTriangles() = 0;
    virtual const SeqQuads& getQuads() = 0;
    virtual const SeqTetras& getTetras() = 0;
    virtual const SeqHexas& getHexas() = 0;
    /// @}

    /// Random accessors
    /// @{
    virtual unsigned int getDOFNumber() const;

    virtual int getNbEdges()     { return getEdges().size(); }
    virtual int getNbTriangles() { return getTriangles().size(); }
    virtual int getNbQuads()     { return getQuads().size(); }
    virtual int getNbTetras()    { return getTetras().size(); }
    virtual int getNbHexas()	 { return getHexas().size(); }

    virtual Edge getEdge(EdgeID i)             { return getEdges()[i]; }
    virtual Triangle getTriangle(TriangleID i) { return getTriangles()[i]; }
    virtual Quad getQuad(QuadID i)             { return getQuads()[i]; }
    virtual Tetra getTetra(TetraID i)          { return getTetras()[i]; }
    virtual Hexa getHexa(HexaID i)             { return getHexas()[i]; }
    /// @}

    /// @name neighbors queries
    /// @{
    /// Returns the set of edges adjacent to a given vertex.
    virtual const VertexEdges& getEdgeVertexShell(PointID i);
    /// Returns the set of edges adjacent to a given triangle.
    virtual const TriangleEdges& getEdgeTriangleShell(TriangleID i);
    /// Returns the set of edges adjacent to a given quad.
    virtual const QuadEdges& getEdgeQuadShell(QuadID i);
    /// Returns the set of edges adjacent to a given tetrahedron.
    virtual const TetraEdges& getEdgeTetraShell(TetraID i);
    /// Returns the set of edges adjacent to a given hexahedron.
    virtual const HexaEdges& getEdgeHexaShell(HexaID i);
    /// Returns the set of triangles adjacent to a given vertex.
    virtual const VertexTriangles& getTriangleVertexShell(PointID i);
    /// Returns the set of triangles adjacent to a given edge.
    virtual const EdgeTriangles& getTriangleEdgeShell(EdgeID i);
    /// Returns the set of triangles adjacent to a given tetrahedron.
    virtual const TetraTriangles& getTriangleTetraShell(TetraID i);
    /// Returns the set of quads adjacent to a given vertex.
    virtual const VertexQuads& getQuadVertexShell(PointID i);
    /// Returns the set of quads adjacent to a given edge.
    virtual const EdgeQuads& getQuadEdgeShell(EdgeID i);
    /// Returns the set of quads adjacent to a given hexahedron.
    virtual const HexaQuads& getQuadHexaShell(HexaID i);
    /// Returns the set of tetrahedra adjacent to a given vertex.
    virtual const VertexTetras& getTetraVertexShell(PointID i);
    /// Returns the set of tetrahedra adjacent to a given edge.
    virtual const EdgeTetras& getTetraEdgeShell(EdgeID i);
    /// Returns the set of tetrahedra adjacent to a given triangle.
    virtual const TriangleTetras& getTetraTriangleShell(TriangleID i);
    /// Returns the set of hexahedra adjacent to a given vertex.
    virtual const VertexHexas& getHexaVertexShell(PointID i);
    /// Returns the set of hexahedra adjacent to a given edge.
    virtual const EdgeHexas& getHexaEdgeShell(EdgeID i);
    /// Returns the set of hexahedra adjacent to a given quad.
    virtual const QuadHexas& getHexaQuadShell(QuadID i);
    /// @}

    /// @name Deprecated names, for backward-compatibility
    /// @{
    const SeqLines& getLines() { return getEdges(); }
    int getNbLines() { return getNbEdges(); }
    Line getLine(LineID i) { return getEdge(i); }
#ifndef SOFA_NEW_HEXA
    const SeqCubes& getCubes() { return getHexas(); }
    int getNbCubes() { return getNbHexas(); }
    Cube getCube(CubeID i) { return getHexa(i); }
#endif
    /// @}

    // Points accessors (not always available)
    virtual bool hasPos() const { return false; }
    virtual double getPX(int) const { return 0.0; }
    virtual double getPY(int) const { return 0.0; }
    virtual double getPZ(int) const { return 0.0; }

    // for procedural creation without file loader
    virtual void clear();
    virtual void addPoint(double px, double py, double pz);
    virtual void addEdge( int a, int b );
    void addLine( int a, int b ) { addEdge(a,b); }
    virtual void addTriangle( int a, int b, int c );
    virtual void addQuad( int a, int b, int c, int d );
    virtual void addTetra( int a, int b, int c, int d );
    virtual void addHexa( int a, int b, int c, int d, int e, int f, int g, int h );

    /// get the current revision of this mesh (use to detect changes)
    /// @deprecated
    virtual int getRevision() const { return 0; }

    /// return true if the given cube is active, i.e. it contains or is surrounded by mapped points.
    /// @deprecated
    virtual bool isCubeActive(int /*index*/) { return true; }

    /** \brief Provides an iterator on the first element in the list of TopologyChange objects.
     */
    virtual std::list<const TopologyChange *>::const_iterator firstChange() const;

    /** \brief Provides an iterator on the last element in the list of TopologyChange objects.
     */
    virtual std::list<const TopologyChange *>::const_iterator lastChange() const;
};

} // namespace topology

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
