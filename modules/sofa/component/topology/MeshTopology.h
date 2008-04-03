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

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;
using helper::vector;
using helper::fixed_array;

class MeshTopology : public core::componentmodel::topology::BaseMeshTopology
{
public:
    MeshTopology();

    virtual void clear();

    virtual bool load(const char* filename);

    virtual int getNbPoints() const;

    // Complete sequence accessors

    virtual const SeqEdges& getEdges();
    virtual const SeqTriangles& getTriangles();
    virtual const SeqQuads& getQuads();
    virtual const SeqTetras& getTetras();
    virtual const SeqHexas& getHexas();

    // Random accessors

    virtual int getNbEdges();
    virtual int getNbTriangles();
    virtual int getNbQuads();
    virtual int getNbTetras();
    virtual int getNbHexas();

    virtual Edge getEdge(EdgeID i);
    virtual Triangle getTriangle(TriangleID i);
    virtual Quad getQuad(QuadID i);
    virtual Tetra getTetra(TetraID i);
    virtual Hexa getHexa(HexaID i);

    /// @name neighbors queries
    /// @{
    /// Returns the set of edges adjacent to a given vertex.
    virtual const vector<EdgeID> &getEdgeVertexShell(PointID i);
    /// Returns the set of edges adjacent to a given triangle.
    virtual const vector<EdgeID> &getEdgeTriangleShell(TriangleID i);
    /// Returns the set of edges adjacent to a given quad.
    virtual const vector<EdgeID> &getEdgeQuadShell(QuadID i);
    /// Returns the set of triangle adjacent to a given vertex.
    virtual const vector<TriangleID> &getTriangleVertexShell(PointID i);
    /// Returns the set of triangle adjacent to a given edge.
    virtual const vector<TriangleID> &getTriangleEdgeShell(EdgeID i);
    /// Returns the set of quad adjacent to a given vertex.
    virtual const vector<QuadID> &getQuadVertexShell(PointID i);
    /// Returns the set of quad adjacent to a given edge.
    virtual const vector<QuadID> &getQuadEdgeShell(EdgeID i);
    /// @}

    // Points accessors (not always available)

    virtual bool hasPos() const;
    virtual double getPX(int i) const;
    virtual double getPY(int i) const;
    virtual double getPZ(int i) const;
    virtual std::string getFilename() const {return filename.getValue();}

    // for procedural creation without file loader
    void addPoint(double px, double py, double pz);
    void addEdge( int a, int b );
    void addTriangle( int a, int b, int c );
    void addTetrahedron( int a, int b, int c, int d );

    // get the current revision of this mesh (use to detect changes)
    int getRevision() const { return revision; }

    /// return true if the given cube is active, i.e. it contains or is surrounded by mapped points.
    /// @deprecated
    virtual bool isCubeActive(int /*index*/) { return true; }

    void parse(core::objectmodel::BaseObjectDescription* arg)
    {
        if (arg->getAttribute("filename"))
        {
            filename.setValue( arg->getAttribute("filename") );
            this->load(arg->getAttribute("filename"));
        }
        arg->removeAttribute("filename");
        this->core::componentmodel::topology::Topology::parse(arg);
    }

protected:
    int nbPoints;
    vector< fixed_array<double,3> > seqPoints;

    Data<SeqEdges> seqEdges;
    bool validEdges;

    //SeqTriangles   seqTriangles;
    Data<SeqTriangles> seqTriangles;
    bool         validTriangles;
    SeqQuads       seqQuads;
    bool         validQuads;

    SeqTetras      seqTetras;
    bool         validTetras;
    SeqCubes       seqHexas;
    bool         validHexas;

    /** the array that stores the set of edge-vertex shells, ie for each vertex gives the set of adjacent edges */
    vector< vector< EdgeID > > m_edgeVertexShell;

    /** the array that stores the set of edge-triangle shells, ie for each triangle gives the set of adjacent edges */
    vector< vector< EdgeID > > m_edgeTriangleShell;

    /// for each vertex provides the set of triangles adjacent to that vertex
    vector< vector< TriangleID > > m_triangleVertexShell;

    /// provides the 3 edges in each triangle
    vector<TriangleEdges> m_triangleEdge;

    /// for each edge provides the set of triangles adjacent to that edge
    vector< vector< TriangleID > > m_triangleEdgeShell;

    /// for each vertex provides the set of quads adjacent to that vertex
    vector< vector< QuadID > > m_quadVertexShell;

    /// for each edge provides the set of quads adjacent to that edge
    vector< vector< QuadID > > m_quadEdgeShell;

    /// provides the 4 edges in each quad
    sofa::helper::vector<QuadEdges> m_quadEdge;

    /** \brief Creates the EdgeSetIndex.
     *
     * This function is only called if the EdgeShell member is required.
     * EdgeShell[i] contains the indices of all edges having the ith DOF as
     * one of their ends.
     */
    void createEdgeVertexShellArray();

    /** \brief Creates the Triangle Vertex Shell Array
     *
     * This function is only called if the TriangleVertexShell array is required.
     * m_triangleVertexShell[i] contains the indices of all triangles adjacent to the ith vertex
     */
    void createTriangleVertexShellArray();

    /** \brief Returns the index of the edge joining vertex v1 and vertex v2; returns -1 if no edge exists
     *
     */
    int getEdgeIndex(PointID v1, PointID v2);

    /** \brief Creates the array of edge indices for each triangle
     *
     * This function is only called if the TriangleEdge array is required.
     * m_triangleEdge[i] contains the 3 indices of the 3 edges opposite to the ith vertex
     */
    void createTriangleEdgeArray();

    /** \brief Returns the TriangleEdges array (ie provide the 3 edge indices for each triangle)
     *
     */
    const vector< TriangleEdges > &getTriangleEdgeArray() ;

    /** \brief Creates the Triangle Edge Shell Array
     *
     * This function is only called if the TriangleVertexShell array is required.
     * m_triangleEdgeShell[i] contains the indices of all triangles adjacent to the ith edge
     */
    void createTriangleEdgeShellArray();

    /** \brief Creates the Quad Vertex Shell Array
     *
     * This function is only called if the QuadVertexShell array is required.
     * m_quadVertexShell[i] contains the indices of all quads adjacent to the ith vertex
     */
    void createQuadVertexShellArray ();

    /** \brief Creates the array of edge indices for each quad
     *
     * This function is only called if the QuadEdge array is required.
     * m_quadEdge[i] contains the 4 indices of the 4 edges opposite to the ith vertex
     */
    void createQuadEdgeArray();

    /** \brief Returns the QuadEdges array (ie provide the 4 edge indices for each quad)
     *
     */
    const vector< QuadEdges > &getQuadEdgeArray();

    /** \brief Creates the Quad Edge Shell Array
     *
     * This function is only called if the QuadVertexShell array is required.
     * m_quadEdgeShell[i] contains the indices of all quads adjacent to the ith edge
     */
    void createQuadEdgeShellArray();


    int revision;

    Data< std::string > filename;

    void invalidate();

    virtual void updateEdges()     { }
    virtual void updateTriangles() { }
    virtual void updateQuads()     { }
    virtual void updateTetras()    { }
    virtual void updateHexas()     { }

    class Loader;
    friend class Loader;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
