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

// CONVENTION : indices ordering for the vertices of an hexahedron :
//
//     Y  3---------2
//     ^ /         /|
//     |/         / |
//     7---------6  |
//     |         |  |
//     |  0------|--1
//     | /       | /
//     |/        |/
//     4---------5-->X
//    /
//   /
//  Z
//
// Hexahedron quads are ordered as {BACK, FRONT, BOTTOM, RIGHT, TOP, LEFT}
// const unsigned int quadsInHexahedronArray[6][4]={{0,1,2,3}, {4,7,6,5}, {1,0,4,5},{1,5,6,2},  {2,6,7,3}, {0,3,7,4}}
// The quads orientation is clockwise
//


#ifndef SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGYCONTAINER_H

#include <sofa/component/topology/QuadSetTopologyContainer.h>


namespace sofa
{
namespace component
{
namespace topology
{
class HexahedronSetTopologyModifier;

using core::topology::BaseMeshTopology;

typedef BaseMeshTopology::PointID			         PointID;
typedef BaseMeshTopology::EdgeID			            EdgeID;
typedef BaseMeshTopology::TriangleID	         	QuadID;
typedef BaseMeshTopology::HexaID			            HexaID;
typedef BaseMeshTopology::Edge				         Edge;
typedef BaseMeshTopology::Quad				         Quad;
typedef BaseMeshTopology::Hexa				         Hexa;
typedef BaseMeshTopology::SeqHexahedra			      SeqHexahedra;
typedef BaseMeshTopology::HexahedraAroundVertex		HexahedraAroundVertex;
typedef BaseMeshTopology::HexahedraAroundEdge		HexahedraAroundEdge;
typedef BaseMeshTopology::HexahedraAroundQuad		HexahedraAroundQuad;
typedef BaseMeshTopology::EdgesInHexahedron		   EdgesInHexahedron;
typedef BaseMeshTopology::QuadsInHexahedron		   QuadsInHexahedron;

typedef Hexa		Hexahedron;
typedef EdgesInHexahedron	EdgesInHexahedron;
typedef QuadsInHexahedron	QuadsInHexahedron;
typedef sofa::helper::vector<HexaID>               VecHexaID;

/** a class that stores a set of hexahedra and provides access with adjacent quads, edges and vertices */
class SOFA_BASE_TOPOLOGY_API HexahedronSetTopologyContainer : public QuadSetTopologyContainer
{
    friend class HexahedronSetTopologyModifier;

public:
    SOFA_CLASS(HexahedronSetTopologyContainer,QuadSetTopologyContainer);

    typedef Hexa		Hexahedron;
    typedef EdgesInHexahedron	EdgesInHexahedron;
    typedef QuadsInHexahedron	QuadsInHexahedron;

    HexahedronSetTopologyContainer();

    virtual ~HexahedronSetTopologyContainer() {}

    virtual void init();


    /// Procedural creation methods
    /// @{
    virtual void clear();
    virtual void addHexa( int a, int b, int c, int d, int e, int f, int g, int h );
    /// @}


    /// BaseMeshTopology API
    /// @{

    /** \brief Returns the hexahedra array. */
    virtual const SeqHexahedra& getHexahedra()
    {
        return getHexahedronArray();
    }

    /** \brief Returns a reference to the Data of hexahedra array container. */
    Data< sofa::helper::vector<Hexahedron> >& getHexahedronDataArray() {return d_hexahedron;}

    /** \brief Returns the ith Hexahedron.
     *
     * @param ID of a hexahedron.
     * @return The corresponding hexahedron.
     */
    virtual const Hexahedron getHexahedron(HexaID i);


    /** \brief Returns the indices of a hexahedron given 8 vertex indices.
     *
     * @param the 8 vertex indices.
     * @return the ID of the corresponding hexahedron.
     * @return -1 if none
     */
    virtual int getHexahedronIndex(PointID v1, PointID v2, PointID v3, PointID v4,
            PointID v5, PointID v6, PointID v7, PointID v8);


    /** \brief Returns the 12 edges adjacent to a given hexahedron.
     *
     * @param ID of a hexahedron.
     * @return EdgesInHexahedron list composing the input hexahedron.
     */
    virtual const EdgesInHexahedron& getEdgesInHexahedron(HexaID i) ;


    /** \brief Returns the 6 quads adjacent to a given hexahedron.
     *
     * @param ID of a hexahedron.
     * @return QuadsInHexahedron list composing the input hexahedron.
     */
    virtual const QuadsInHexahedron& getQuadsInHexahedron(HexaID i) ;


    /** \brief Returns the set of hexahedra adjacent to a given vertex.
     *
     * @param ID of a vertex.
     * @return HexahedraAroundVertex list around the input vertex.
     */
    virtual const HexahedraAroundVertex& getHexahedraAroundVertex(PointID i) ;


    /** \brief Returns the set of hexahedra adjacent to a given edge.
     *
     * @param ID of a edge.
     * @return HexahedraAroundEdge list around the input edge.
     */
    virtual const HexahedraAroundEdge& getHexahedraAroundEdge(EdgeID i) ;


    /** \brief Returns the set of hexahedra adjacent to a given quad.
     *
     * @param ID of a quad.
     * @return HexahedraAroundQuad list around the input quad.
     */
    virtual const HexahedraAroundQuad& getHexahedraAroundQuad(QuadID i) ;


    /** returns the index (0 to 7) of the vertex whose global index is vertexIndex.
     *
     * @param Ref to a Hexahedron.
     * @param Id of a vertex.
     * @return the position of this vertex in the Hexahedron (i.e. either 0, 1, ..., 7).
     * @return -1 if none.
     */
    int getVertexIndexInHexahedron(const Hexahedron &t, PointID vertexIndex) const;


    /** returns the index (0 to 11) of the edge whose global index is edgeIndex.
     *
     * @param Ref to a EdgesInHexahedron.
     * @param Id of an edge.
     * @return the position of this edge in the Hexahedron (i.e. either 0, 1, ..., 11).
     * @return -1 if none.
     */
    int getEdgeIndexInHexahedron(const EdgesInHexahedron &t, EdgeID edgeIndex) const;


    /** returns the index (0 to 7) of the quad whose global index is quadIndex.
     *
     * @param Ref to a QuadsInHexahedron.
     * @param Id of a quad.
     * @return the position of this quad in the Hexahedron (i.e. either 0, 1, ..., 7).
     * @return -1 if none.
     */
    int getQuadIndexInHexahedron(const QuadsInHexahedron &t, QuadID quadIndex) const;


    /** \brief Returns for each index (between 0 and 11) the two vertex local indices that are adjacent to/forming that edge
     *
     */
    virtual Edge getLocalEdgesInHexahedron (const EdgeID i) const;


    /** \brief Returns for each index (between 0 and 5) the four vertices local indices that are adjacent to/forming that quad
     *
     */
    virtual Quad getLocalQuadsInHexahedron (const QuadID i) const;

    /** \brief Given an EdgesInQuad and a QuadsInHexahedron index in a hexahedron, returns the QuadsInHexahedron index of the quad sharing the same edge.
     *
     */
    virtual QuadID getNextAdjacentQuad(const HexaID _hexaID, const QuadID _quadID, const EdgeID _edgeID);

    /// @}



    /// Dynamic Topology API
    /// @{

    /** \brief Checks if the topology is coherent
     *
     * Check if the shell arrays are coherent
     * @see m_hexahedron
     * @see m_edgesInHexahedron
     * @see m_quadsInHexahedron
     * @see m_hexahedraAroundVertex
     * @see m_hexahedraAroundEdge
     * @see m_hexahedraAroundQuad
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
    virtual const VecHexaID getConnectedElement(HexaID elem);

    /// Returns the set of element indices adjacent to a given element (i.e. sharing a link)
    virtual const VecHexaID getElementAroundElement(HexaID elem);
    /// Returns the set of element indices adjacent to a given list of elements (i.e. sharing a link)
    virtual const VecHexaID getElementAroundElements(VecHexaID elems);
    /// @}


    /** \brief Returns the number of hexahedra in this topology.
     *	The difference to getNbHexahedra() is that this method does not generate the hexa array if it does not exist.
     */
    unsigned int getNumberOfHexahedra() const;


    /** \brief Returns the number of topological element of the current topology.
     * This function avoids to know which topological container is in used.
     */
    virtual unsigned int getNumberOfElements() const;


    /** \brief Returns the Hexahedron array. */
    const sofa::helper::vector<Hexahedron> &getHexahedronArray();


    /** \brief Returns the EdgesInHexahedron array (i.e. provide the 12 edge indices for each hexahedron).	*/
    const sofa::helper::vector< EdgesInHexahedron > &getEdgesInHexahedronArray() ;


    /** \brief Returns the QuadsInHexahedron array (i.e. provide the 8 quad indices for each hexahedron).	*/
    const sofa::helper::vector< QuadsInHexahedron > &getQuadsInHexahedronArray() ;


    /** \brief Returns the HexahedraAroundVertex array (i.e. provide the hexahedron indices adjacent to each vertex).*/
    const sofa::helper::vector< HexahedraAroundVertex > &getHexahedraAroundVertexArray() ;


    /** \brief Returns the HexahedraAroundEdge array (i.e. provide the hexahedron indices adjacent to each edge). */
    const sofa::helper::vector< HexahedraAroundEdge > &getHexahedraAroundEdgeArray() ;


    /** \brief Returns the HexahedraAroundQuad array (i.e. provide the hexahedron indices adjacent to each quad). */
    const sofa::helper::vector< HexahedraAroundQuad > &getHexahedraAroundQuadArray() ;


    bool hasHexahedra() const;

    bool hasEdgesInHexahedron() const;

    bool hasQuadsInHexahedron() const;

    bool hasHexahedraAroundVertex() const;

    bool hasHexahedraAroundEdge() const;

    bool hasHexahedraAroundQuad() const;

    /// @}


protected:

    /** \brief Creates the EdgeSet array.
     *
     * Create the set of edges when needed.
     */
    virtual void createEdgeSetArray();


    /** \brief Creates the QuadSet array.
     *
     * Create the array of quads when needed.
     */
    virtual void createQuadSetArray();


    /** \brief Creates the HexahedronSet array.
     *
     * This function must be implemented by a derived classes.
     */
    virtual void createHexahedronSetArray();


    /** \brief Creates the array of edge indices for each hexahedron.
    *
    * This function is only called if the EdgesInHexahedron array is required.
    * m_edgesInHexahedron[i] contains the 12 indices of the 12 edges of each hexahedron.
    */
    virtual void createEdgesInHexahedronArray();


    /** \brief Creates the array of quad indices for each hexahedron.
    *
    * This function is only called if the QuadsInHexahedron array is required.
    * m_quadsInHexahedron[i] contains the 6 indices of the 6 quads of each hexahedron.
    */
    virtual void createQuadsInHexahedronArray();


    /** \brief Creates the HexahedraAroundVertex Array.
    *
    * This function is only called if the HexahedraAroundVertex array is required.
    * m_hexahedraAroundVertex[i] contains the indices of all hexahedra adjacent to the ith vertex.
    */
    virtual void createHexahedraAroundVertexArray();


    /** \brief Creates the HexahedraAroundEdge Array.
    *
    * This function is only called if the HexahedraAroundEdge array is required.
    * m_hexahedraAroundEdge[i] contains the indices of all hexahedra adjacent to the ith edge.
    */
    virtual void createHexahedraAroundEdgeArray();


    /** \brief Creates the HexahedraAroundQuad Array.
    *
    * This function is only called if the HexahedraAroundQuad array is required.
    * m_hexahedraAroundQuad[i] contains the indices of all hexahedra adjacent to the ith quad.
    */
    virtual void createHexahedraAroundQuadArray();


    void clearHexahedra();

    void clearEdgesInHexahedron();

    void clearQuadsInHexahedron();

    void clearHexahedraAroundVertex();

    void clearHexahedraAroundEdge();

    void clearHexahedraAroundQuad();

protected:


    /** \brief Returns a non-const list of hexahedron indices around a given DOF for subsequent modification.
     *
     * @return HexahedraAroundVertex lists in non-const.
     * @see getHexahedraAroundVertex()
     */
    virtual HexahedraAroundVertex& getHexahedraAroundVertexForModification(const PointID vertexIndex);


    /** \brief Returns a non-const list of hexahedron indices around a given edge for subsequent modification.
     *
     * @return HexahedraAroundEdge lists in non-const.
     * @see getHexahedraAroundEdge()
     */
    virtual HexahedraAroundEdge& getHexahedraAroundEdgeForModification(const EdgeID edgeIndex);


    /** \brief Returns a non-const list of hexahedron indices around a given quad for subsequent modification.
     *
     * @return HexahedraAroundQuad lists in non-const.
     * @see getHexahedraAroundQuad()
     */
    virtual HexahedraAroundQuad& getHexahedraAroundQuadForModification(const QuadID quadIndex);


protected:

    /// provides the set of hexahedra.
    Data< sofa::helper::vector<Hexahedron> > d_hexahedron;

    /// provides the set of edges for each hexahedron.
    sofa::helper::vector<EdgesInHexahedron> m_edgesInHexahedron;

    /// provides the set of quads for each hexahedron.
    sofa::helper::vector<QuadsInHexahedron> m_quadsInHexahedron;

    /// for each vertex provides the set of hexahedra adjacent to that vertex.
    sofa::helper::vector< HexahedraAroundVertex > m_hexahedraAroundVertex;

    /// for each edge provides the set of hexahedra adjacent to that edge.
    sofa::helper::vector< HexahedraAroundEdge > m_hexahedraAroundEdge;

    /// for each quad provides the set of hexahedra adjacent to that quad.
    sofa::helper::vector< HexahedraAroundQuad > m_hexahedraAroundQuad;

};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
