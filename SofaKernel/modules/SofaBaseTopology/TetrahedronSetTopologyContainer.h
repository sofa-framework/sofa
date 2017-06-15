/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGYCONTAINER_H
#include "config.h"

#include <SofaBaseTopology/TriangleSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace topology
{
class TetrahedronSetTopologyModifier;



/** a class that stores a set of tetrahedra and provides access with adjacent triangles, edges and vertices */
class SOFA_BASE_TOPOLOGY_API TetrahedronSetTopologyContainer : public TriangleSetTopologyContainer
{
    friend class TetrahedronSetTopologyModifier;

public:
    SOFA_CLASS(TetrahedronSetTopologyContainer,TriangleSetTopologyContainer);


    typedef core::topology::BaseMeshTopology::PointID			         PointID;
    typedef core::topology::BaseMeshTopology::EdgeID			            EdgeID;
    typedef core::topology::BaseMeshTopology::TriangleID		         TriangleID;
    typedef core::topology::BaseMeshTopology::TetraID			         TetraID;
    typedef core::topology::BaseMeshTopology::Edge				         Edge;
    typedef core::topology::BaseMeshTopology::Triangle			         Triangle;
    typedef core::topology::BaseMeshTopology::Tetra				         Tetra;
    typedef core::topology::BaseMeshTopology::SeqTetrahedra			   SeqTetrahedra;
    typedef core::topology::BaseMeshTopology::TetrahedraAroundVertex	TetrahedraAroundVertex;
    typedef core::topology::BaseMeshTopology::TetrahedraAroundEdge		TetrahedraAroundEdge;
    typedef core::topology::BaseMeshTopology::TetrahedraAroundTriangle	TetrahedraAroundTriangle;
    typedef core::topology::BaseMeshTopology::EdgesInTetrahedron		   EdgesInTetrahedron;
    typedef core::topology::BaseMeshTopology::TrianglesInTetrahedron	TrianglesInTetrahedron;


    typedef Tetra			Tetrahedron;
    typedef sofa::helper::vector<TetraID>         VecTetraID;

protected:
    TetrahedronSetTopologyContainer();

    virtual ~TetrahedronSetTopologyContainer() {}
public:
    virtual void init();

    //add removed tetrahedron index
    void addRemovedTetraIndex(sofa::helper::vector< unsigned int >& tetrahedra);

    //get removed tetrahedron index
    sofa::helper::vector< unsigned int >& getRemovedTetraIndex();

    /// Procedural creation methods
    /// @{
    virtual void clear();
    virtual void addTriangle( int, int, int ) {}
    virtual void addTetra( int a, int b, int c, int d );
    /// @}



    /// BaseMeshTopology API
    /// @{

    /** \brief Returns the tetrahedra array. */
    virtual const SeqTetrahedra& getTetrahedra()
    {
        return getTetrahedronArray();
    }

    /** \brief Returns a reference to the Data of tetrahedra array container. */
    Data< sofa::helper::vector<Tetrahedron> >& getTetrahedronDataArray() {return d_tetrahedron;}

    /** \brief Returns the tetrahedron corresponding to the TetraID i.
     *
     * @param ID of a tetrahedron.
     * @return The corresponding tetrahderon.
     */
    virtual const Tetrahedron getTetrahedron (TetraID i);


    /** \brief Returns the indices of a tetrahedron given four vertex indices.
     *
     * @param the four vertex indices.
     * @return the ID of the corresponding tetrahedron.
     * @return -1 if none
     */
    virtual int getTetrahedronIndex(PointID v1, PointID v2, PointID v3, PointID v4);


    /** \brief Returns the 6 edges adjacent to a given tetrahedron.
     *
     * @param ID of a tetrahedron.
     * @return EdgesInTetrahedron list composing the input tetrahedron.
     */
    virtual const EdgesInTetrahedron& getEdgesInTetrahedron(TetraID i) ;


    /** \brief Returns the 4 triangles adjacent to a given tetrahedron.
     *
     * @param ID of a tetrahedron.
     * @return TrianglesInTetrahedron list composing the input tetrahedron.
     */
    virtual const TrianglesInTetrahedron& getTrianglesInTetrahedron(TetraID i) ;


    /** \brief Returns the set of tetrahedra adjacent to a given vertex.
     *
     * @param ID of a vertex.
     * @return TetrahedraAroundVertex list around the input vertex.
     */
    virtual const TetrahedraAroundVertex& getTetrahedraAroundVertex(PointID i);


    /** \brief Returns the set of tetrahedra adjacent to a given edge.
     *
     * @param ID of an edge.
     * @return TetrahedraAroundVertex list around the input edge.
     */
    virtual const TetrahedraAroundEdge& getTetrahedraAroundEdge(EdgeID i) ;


    /** \brief Returns the set of tetrahedra adjacent to a given triangle.
     *
     * @param ID of a triangle.
     * @return TetrahedraAroundVertex list around the input triangle.
     */
    virtual const TetrahedraAroundTriangle& getTetrahedraAroundTriangle(TriangleID i) ;


    /** \brief Returns the index (either 0, 1 ,2 or 3) of the vertex whose global index is vertexIndex.
     *
     * @param Ref to a Tetrahedron.
     * @param Id of a vertex.
     * @return the position of this vertex in the tetrahedron (i.e. either 0, 1, 2 or 3).
     * @return -1 if none.
     */
    virtual int getVertexIndexInTetrahedron(const Tetrahedron &t, PointID vertexIndex) const;


    /** \brief Returns the index (either 0, 1 ,2, 3, 4 or 5) of the edge whose global index is edgeIndex.
     *
     * @param Ref to an EdgesInTetrahedron.
     * @param Id of an edge.
     * @return the position of this edge in the tetrahedron (i.e. either 0, 1, 2, 3, 4 or 5).
     * @return -1 if none.
     */
    virtual int getEdgeIndexInTetrahedron(const EdgesInTetrahedron &t, EdgeID edgeIndex) const;


    /** \brief Returns the index (either 0, 1 ,2 or 3) of the triangle whose global index is triangleIndex.
     *
     * @param Ref to a TrianglesInTetrahedron.
     * @param Id of a triangle.
     * @return the position of this triangle in the tetrahedron (i.e. either 0, 1, 2 or 3).
     * @return -1 if none.
     */
    virtual int getTriangleIndexInTetrahedron(const TrianglesInTetrahedron &t, TriangleID triangleIndex) const;


    /** \brief Returns for each index (between 0 and 5) the two vertex indices that are adjacent to that edge.
     *
     */
    virtual Edge getLocalEdgesInTetrahedron (const unsigned int i) const;


    /** \brief Returns for each index (between 0 and 3) the three local vertices indices that are adjacent to that triangle
     *
     */
    virtual Triangle getLocalTrianglesInTetrahedron (const PointID i) const;

    /// @}



    /// Dynamic Topology API
    /// @{

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
    virtual bool checkTopology() const;


    /// Get information about connexity of the mesh
    /// @{
    /** \brief Checks if the topology has only one connected component
      *
      * @return true if only one connected component
      */
    virtual bool checkConnexity();

    /// Returns the number of connected component.
    virtual unsigned int getNumberOfConnectedComponent();

    /// Returns the set of element indices connected to an input one (i.e. which can be reached by topological links)
    virtual const VecTetraID getConnectedElement(TetraID elem);

    /// Returns the set of element indices adjacent to a given element (i.e. sharing a link)
    virtual const VecTetraID getElementAroundElement(TetraID elem);
    /// Returns the set of element indices adjacent to a given list of elements (i.e. sharing a link)
    virtual const VecTetraID getElementAroundElements(VecTetraID elems);
    /// @}



    /** \brief Returns the number of tetrahedra in this topology.
     *	The difference to getNbTetrahedra() is that this method does not generate the tetra array if it does not exist.
     */
    unsigned int getNumberOfTetrahedra() const;

    /** \brief Returns the number of topological element of the current topology.
     * This function avoids to know which topological container is in used.
     */
    virtual unsigned int getNumberOfElements() const;


    /** \brief Returns the Tetrahedron array. */
    const sofa::helper::vector<Tetrahedron> &getTetrahedronArray();


    /** \brief Returns the EdgesInTetrahedron array (i.e. provide the 6 edge indices for each tetrahedron). */
    const sofa::helper::vector< EdgesInTetrahedron > &getEdgesInTetrahedronArray() ;


    /** \brief Returns the TrianglesInTetrahedron array (i.e. provide the 4 triangle indices for each tetrahedron). */
    const sofa::helper::vector< TrianglesInTetrahedron > &getTrianglesInTetrahedronArray() ;


    /** \brief Returns the TetrahedraAroundVertex array (i.e. provide the tetrahedron indices adjacent to each vertex). */
    const sofa::helper::vector< TetrahedraAroundVertex > &getTetrahedraAroundVertexArray() ;


    /** \brief Returns the TetrahedraAroundEdge array (i.e. provide the tetrahedron indices adjacent to each edge). */
    const sofa::helper::vector< TetrahedraAroundEdge > &getTetrahedraAroundEdgeArray() ;


    /** \brief Returns the TetrahedraAroundTriangle array (i.e. provide the tetrahedron indices adjacent to each triangle). */
    const sofa::helper::vector< TetrahedraAroundTriangle > &getTetrahedraAroundTriangleArray() ;


    bool hasTetrahedra() const;

    bool hasEdgesInTetrahedron() const;

    bool hasTrianglesInTetrahedron() const;

    bool hasTetrahedraAroundVertex() const;

    bool hasTetrahedraAroundEdge() const;

    bool hasTetrahedraAroundTriangle() const;

    /// @}


    inline friend std::ostream& operator<< (std::ostream& out, const TetrahedronSetTopologyContainer& t)
    {
        helper::ReadAccessor< Data< sofa::helper::vector<Tetrahedron> > > m_tetrahedron = t.d_tetrahedron;
        out  << m_tetrahedron<< " "
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
        unsigned int s=0;
        sofa::helper::vector< unsigned int > value;
        helper::WriteAccessor< Data< sofa::helper::vector<Tetrahedron> > > m_tetrahedron = t.d_tetrahedron;

        in >> m_tetrahedron >> t.m_edgesInTetrahedron >> t.m_trianglesInTetrahedron;


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


    /// \brief Function creating the data graph linked to d_tetrahedron
    virtual void updateTopologyEngineGraph();


    /// Use a specific boolean @see m_tetrahedronTopologyDirty in order to know if topology Data is dirty or not.
    /// Set/Get function access to this boolean
    void setTetrahedronTopologyToDirty() {m_tetrahedronTopologyDirty = true;}
    void cleanTetrahedronTopologyFromDirty() {m_tetrahedronTopologyDirty = false;}
    const bool& isTetrahedronTopologyDirty() {return m_tetrahedronTopologyDirty;}

public:
	/// force the creation of triangles
	Data<bool>  d_createTriangleArray;

    /// provides the set of tetrahedra.
    Data< sofa::helper::vector<Tetrahedron> > d_tetrahedron;
protected:
    /// provides the set of edges for each tetrahedron.
    sofa::helper::vector<EdgesInTetrahedron> m_edgesInTetrahedron;

    /// provides the set of triangles for each tetrahedron.
    sofa::helper::vector<TrianglesInTetrahedron> m_trianglesInTetrahedron;

    /// for each vertex provides the set of tetrahedra adjacent to that vertex.
    sofa::helper::vector< TetrahedraAroundVertex > m_tetrahedraAroundVertex;

    /// for each edge provides the set of tetrahedra adjacent to that edge.
    sofa::helper::vector< TetrahedraAroundEdge > m_tetrahedraAroundEdge;

    /// removed tetrahedron index
    sofa::helper::vector<unsigned int> m_removedTetraIndex;

    /// for each triangle provides the set of tetrahedra adjacent to that triangle.
    sofa::helper::vector< TetrahedraAroundTriangle > m_tetrahedraAroundTriangle;


    /// Boolean used to know if the topology Data of this container is dirty
    bool m_tetrahedronTopologyDirty;

    /// List of engines related to this specific container
    std::list<sofa::core::topology::TopologyEngine *> m_enginesList;

    /// \brief variables used to display the graph of Data/DataEngines linked to this Data array.
    sofa::helper::vector < sofa::helper::vector <std::string> > m_dataGraph;
    sofa::helper::vector < sofa::helper::vector <std::string> > m_enginesGraph;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
