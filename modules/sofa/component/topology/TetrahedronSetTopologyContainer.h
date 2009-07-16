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
#ifndef SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGYCONTAINER_H

#include <sofa/component/topology/TriangleSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace topology
{
class TetrahedronSetTopologyModifier;

using core::componentmodel::topology::BaseMeshTopology;

typedef BaseMeshTopology::PointID			PointID;
typedef BaseMeshTopology::EdgeID			EdgeID;
typedef BaseMeshTopology::TriangleID		        TriangleID;
typedef BaseMeshTopology::TetraID			TetraID;
typedef BaseMeshTopology::Edge				Edge;
typedef BaseMeshTopology::Triangle			Triangle;
typedef BaseMeshTopology::Tetra				Tetra;
typedef BaseMeshTopology::SeqTetrahedra			SeqTetrahedra;
typedef BaseMeshTopology::TetrahedraAroundVertex	TetrahedraAroundVertex;
typedef BaseMeshTopology::TetrahedraAroundEdge		TetrahedraAroundEdge;
typedef BaseMeshTopology::TetrahedraAroundTriangle	TetrahedraAroundTriangle;
typedef BaseMeshTopology::EdgesInTetrahedron		EdgesInTetrahedron;
typedef BaseMeshTopology::TrianglesInTetrahedron	TrianglesInTetrahedron;

typedef Tetra			Tetrahedron;
typedef EdgesInTetrahedron		EdgesInTetrahedron;
typedef TrianglesInTetrahedron	TrianglesInTetrahedron;

/** a class that stores a set of tetrahedra and provides access with adjacent triangles, edges and vertices */
class SOFA_COMPONENT_CONTAINER_API TetrahedronSetTopologyContainer : public TriangleSetTopologyContainer
{
    friend class TetrahedronSetTopologyModifier;

public:
    typedef Tetra			Tetrahedron;
    typedef EdgesInTetrahedron		EdgesInTetrahedron;
    typedef TrianglesInTetrahedron	TrianglesInTetrahedron;


    TetrahedronSetTopologyContainer();

    TetrahedronSetTopologyContainer(const sofa::helper::vector< Tetrahedron >& tetrahedra );

    virtual ~TetrahedronSetTopologyContainer() {}

    /// Procedural creation methods
    /// @{
    virtual void clear();
    virtual void addTriangle( int, int, int ) {}
    virtual void addTetra( int a, int b, int c, int d );
    /// @}

    virtual void init();

    /// BaseMeshTopology API
    /// @{

    const SeqTetrahedra& getTetrahedra()
    {
        return getTetrahedronArray();
    }


    const Tetrahedron getTetrahedron (TetraID i);


    /*
    /// Returns the set of edges adjacent to a given tetrahedron.
    const EdgesInTetrahedron& getEdgesInTetrahedron(TetraID i)
    {
    	return getEdgesInTetrahedron(i);
    }

    /// Returns the set of triangles adjacent to a given tetrahedron.
    const TrianglesInTetrahedron& getTrianglesInTetrahedron(TetraID i)
    {
    	return getTrianglesInTetrahedron(i);
    }

    /// Returns the set of tetrahedra adjacent to a given vertex.
    const TetrahedraAroundVertex& getTetrahedraAroundVertex(PointID i)
    {
    	return getTetrahedraAroundVertex(i);
    }

    /// Returns the set of tetrahedra adjacent to a given edge.
    const TetrahedraAroundEdge& getTetrahedraAroundEdge(EdgeID i)
    {
    	return getTetrahedraAroundEdge(i);
    }

    /// Returns the set of tetrahedra adjacent to a given triangle.
    const TetrahedraAroundTriangle& getTetrahedraAroundTriangle(TriangleID i)
    {
    	return getTetrahedraAroundTriangle(i);
    }
    */
    /** Returns the indices of a tetrahedron given four vertex indices : returns -1 if none */
    virtual int getTetrahedronIndex(PointID v1, PointID v2, PointID v3, PointID v4);

    /** \brief Returns for each index (between 0 and 5) the two vertex indices that are adjacent to that edge
    *
    */
    virtual Edge getLocalEdgesInTetrahedron (const unsigned int i) const;

    /// @}

    /** \brief Checks if the topology is coherent
    *
    * Check if the shell arrays are coherent
    */
    virtual bool checkTopology() const;

    /** \brief Returns the number of tetrahedra in this topology.
    *	The difference to getNbTetrahedra() is that this method does not generate the tetra array if it does not exist.
    */
    unsigned int getNumberOfTetrahedra() const;

    /** \brief Returns the Tetrahedron array.
    *
    */
    const sofa::helper::vector<Tetrahedron> &getTetrahedronArray();

    /** \brief Returns the Tetrahedron Vertex Shells array.
    *
    */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getTetrahedraAroundVertexArray() ;

    /** \brief Returns the set of tetrahedra adjacent to a given vertex.
    *
    */
    const TetrahedraAroundVertex &getTetrahedraAroundVertex(PointID i);

    /** \brief Returns the Tetrahedron Edges  array.
    *
    */
    const sofa::helper::vector< EdgesInTetrahedron > &getTetrahedronEdgeArray() ;

    /** \brief Returns the 6 edges adjacent to a given tetrahedron.
    *
    */
    const EdgesInTetrahedron &getEdgesInTetrahedron(TetraID i) ;

    /** \brief Returns the Tetrahedron Triangles  array.
    *
    */
    const sofa::helper::vector< TrianglesInTetrahedron > &getTetrahedronTriangleArray() ;

    /** \brief Returns the 4 triangles adjacent to a given tetrahedron.
    *
    */
    const TrianglesInTetrahedron &getTrianglesInTetrahedron(TetraID i) ;

    /** \brief Returns the Tetrahedron Edge Shells array.
    *
    */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getTetrahedraAroundEdgeArray() ;

    /** \brief Returns the set of tetrahedra adjacent to a given edge.
    *
    */
    const TetrahedraAroundEdge &getTetrahedraAroundEdge(EdgeID i) ;

    /** \brief Returns the Tetrahedron Triangle Shells array.
    *
    */
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getTetrahedraAroundTriangleArray() ;

    /** \brief Returns the set of tetrahedra adjacent to a given triangle.
    *
    */
    const TetrahedraAroundTriangle &getTetrahedraAroundTriangle(TriangleID i) ;

    /** returns the index (either 0, 1 ,2 or 3) of the vertex whose global index is vertexIndex. Returns -1 if none */
    int getVertexIndexInTetrahedron(const Tetrahedron &t,unsigned int vertexIndex) const;

    /** returns the index (either 0, 1 ,2, 3, 4 or 5) of the edge whose global index is edgeIndex. Returns -1 if none */
    int getEdgeIndexInTetrahedron(const EdgesInTetrahedron &t,unsigned int edgeIndex) const;

    /** returns the index (either 0, 1 ,2 or 3) of the triangle whose global index is triangleIndex. Returns -1 if none */
    int getTriangleIndexInTetrahedron(const TrianglesInTetrahedron &t,unsigned int triangleIndex) const;

    inline friend std::ostream& operator<< (std::ostream& out, const TetrahedronSetTopologyContainer& t)
    {
        out  << t.m_tetrahedron<< " "
                << t.m_edgesInTetrahedron<< " "
                << t.m_trianglesInTetrahedron;

        out << " "<< t.m_tetrahedraAroundVertex.size();
        for (unsigned int i=0; i<t.m_tetrahedraAroundVertex.size(); i++)
        {
            out << " " << t.m_tetrahedraAroundVertex[i];
        }
        out <<" "<< t.m_tetrahedraAroundEdge.size();
        for (unsigned int i=0; i<t.m_tetrahedraAroundEdge.size(); i++)
        {
            out << " " << t.m_tetrahedraAroundEdge[i];
        }
        out <<" "<< t.m_tetrahedraAroundTriangle.size();
        for (unsigned int i=0; i<t.m_tetrahedraAroundTriangle.size(); i++)
        {
            out << " " << t.m_tetrahedraAroundTriangle[i];
        }
        return out;
    }

    inline friend std::istream& operator>>(std::istream& in, TetrahedronSetTopologyContainer& t)
    {
        unsigned int s;
        sofa::helper::vector< unsigned int > value;


        in >> t.m_tetrahedron >> t.m_edgesInTetrahedron >> t.m_trianglesInTetrahedron;


        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            in >> value;
            t.m_tetrahedraAroundVertex.push_back(value);
        }
        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            in >> value;
            t.m_tetrahedraAroundEdge.push_back(value);
        }
        in >> s;
        for (unsigned int i=0; i<s; i++)
        {
            in >> value;
            t.m_tetrahedraAroundTriangle.push_back(value);
        }
        return in;
    }

protected:
    /** \brief Creates the EdgeSet array.
    *
    * Create the set of edges when needed.
    */
    virtual void createEdgeSetArray();

    /** \brief Creates the TriangleSet array.
    *
    * Create the array of Triangles
    */
    virtual void createTriangleSetArray();

    /** \brief Creates the TetrahedronSet array.
    *
    * This function must be implemented by a derived classes
    */
    virtual void createTetrahedronSetArray();

    bool hasTetrahedra() const;

    bool hasEdgesInTetrahedron() const;

    bool hasTrianglesInTetrahedron() const;

    bool hasTetrahedraAroundVertex() const;

    bool hasTetrahedraAroundEdge() const;

    bool hasTetrahedraAroundTriangle() const;

    void clearTetrahedra();

    void clearEdgesInTetrahedron();

    void clearTrianglesInTetrahedron();

    void clearTetrahedraAroundVertex();

    void clearTetrahedraAroundEdge();

    void clearTetrahedraAroundTriangle();

private:
    /** \brief Creates the array of edge indices for each tetrahedron
    *
    * This function is only called if the TetrahedronEdge array is required.
    * m_edgesInTetrahedron[i] contains the 6 indices of the 6 edges of each tetrahedron
    The number of each edge is the following : edge 0 links vertex 0 and 1, edge 1 links vertex 0 and 2,
    edge 2 links vertex 0 and 3, edge 3 links vertex 1 and 2, edge 4 links vertex 1 and 3,
    edge 5 links vertex 2 and 3
    */
    void createTetrahedronEdgeArray();

    /** \brief Creates the array of triangle indices for each tetrahedron
    *
    * This function is only called if the TetrahedronTriangle array is required.
    * m_trianglesInTetrahedron[i] contains the 4 indices of the 4 triangles opposite to the ith vertex
    */
    void createTetrahedronTriangleArray();

protected:
    /// provides the set of tetrahedra
    sofa::helper::vector<Tetrahedron> m_tetrahedron;
    DataPtr< sofa::helper::vector<Tetrahedron> > d_tetrahedron;
    /// provides the set of edges for each tetrahedron
    sofa::helper::vector<EdgesInTetrahedron> m_edgesInTetrahedron;
    /// provides the set of triangles for each tetrahedron
    sofa::helper::vector<TrianglesInTetrahedron> m_trianglesInTetrahedron;

    /// for each vertex provides the set of tetrahedra adjacent to that vertex
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_tetrahedraAroundVertex;
    /// for each edge provides the set of tetrahedra adjacent to that edge
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_tetrahedraAroundEdge;
    /// for each triangle provides the set of tetrahedra adjacent to that edge
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_tetrahedraAroundTriangle;

    virtual void loadFromMeshLoader(sofa::component::container::MeshLoader* loader);

    /** \brief Creates the Tetrahedron Vertex Shell Array
    *
    * This function is only called if the TetrahedraAroundVertex array is required.
    * m_tetrahedraAroundVertex[i] contains the indices of all tetrahedra adjacent to the ith vertex
    */
    virtual void createTetrahedraAroundVertexArray();

    /** \brief Creates the Tetrahedron Edge Shell Array
    *
    * This function is only called if the TetrahedronEdheShell array is required.
    * m_tetrahedraAroundEdge[i] contains the indices of all tetrahedra adjacent to the ith edge
    */
    virtual void createTetrahedraAroundEdgeArray();

    /** \brief Creates the Tetrahedron Triangle Shell Array
    *
    * This function is only called if the TetrahedraAroundTriangle array is required.
    * m_tetrahedraAroundTriangle[i] contains the indices of all tetrahedra adjacent to the ith edge
    */
    virtual void createTetrahedraAroundTriangleArray();

    /** \brief Returns a non-const tetrahedron vertex shell given a vertex index for subsequent modification
    *
    */
    sofa::helper::vector< unsigned int > &getTetrahedraAroundVertexForModification(const unsigned int vertexIndex);

    /** \brief Returns a non-const tetrahedron edge shell given the index of an edge for subsequent modification
    *
    */
    sofa::helper::vector< unsigned int > &getTetrahedraAroundEdgeForModification(const unsigned int edgeIndex);


    sofa::helper::vector< unsigned int > &getTetrahedraAroundTriangleForModification(const unsigned int triangleIndex);


};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
