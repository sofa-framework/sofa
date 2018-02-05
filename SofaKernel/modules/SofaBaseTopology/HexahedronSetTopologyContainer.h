/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#include <SofaBaseTopology/QuadSetTopologyContainer.h>


namespace sofa
{
namespace component
{
namespace topology
{
class HexahedronSetTopologyModifier;


/** a class that stores a set of hexahedra and provides access with adjacent quads, edges and vertices */
class SOFA_BASE_TOPOLOGY_API HexahedronSetTopologyContainer : public QuadSetTopologyContainer
{
    friend class HexahedronSetTopologyModifier;

public:
    SOFA_CLASS(HexahedronSetTopologyContainer,QuadSetTopologyContainer);



    typedef core::topology::BaseMeshTopology::PointID			         PointID;
    typedef core::topology::BaseMeshTopology::PointID			         LocalPointID;
    typedef core::topology::BaseMeshTopology::EdgeID			            EdgeID;
    typedef core::topology::BaseMeshTopology::TriangleID	         	QuadID;
    typedef core::topology::BaseMeshTopology::HexaID			            HexaID;
    typedef core::topology::BaseMeshTopology::Edge				         Edge;
    typedef core::topology::BaseMeshTopology::Quad				         Quad;
    typedef core::topology::BaseMeshTopology::Hexa				         Hexa;
    typedef core::topology::BaseMeshTopology::SeqHexahedra			      SeqHexahedra;
    typedef core::topology::BaseMeshTopology::HexahedraAroundVertex		HexahedraAroundVertex;
    typedef core::topology::BaseMeshTopology::HexahedraAroundEdge		HexahedraAroundEdge;
    typedef core::topology::BaseMeshTopology::HexahedraAroundQuad		HexahedraAroundQuad;
    typedef core::topology::BaseMeshTopology::EdgesInHexahedron		   EdgesInHexahedron;
    typedef core::topology::BaseMeshTopology::QuadsInHexahedron		   QuadsInHexahedron;

    typedef sofa::helper::vector<HexaID>               VecHexaID;


    typedef Hexa		Hexahedron;
	typedef sofa::defaulttype::Vec<3,unsigned char> HexahedronBinaryIndex;
protected:
    HexahedronSetTopologyContainer();

    virtual ~HexahedronSetTopologyContainer() override {}
public:
    virtual void init() override;


    /// Procedural creation methods
    /// @{
    virtual void clear() override;
    virtual void addHexa( int a, int b, int c, int d, int e, int f, int g, int h ) override;
    /// @}


    /// BaseMeshTopology API
    /// @{

    /** \brief Get the array of hexahedra. */
    virtual const SeqHexahedra& getHexahedra() override
    {
        return getHexahedronArray();
    }

    /** \brief Get the Data which contains the array of hexahedra. */
    Data< sofa::helper::vector<Hexahedron> >& getHexahedronDataArray() {return d_hexahedron;}

    /** \brief Get a hexahedron from its index.
     *
     * @param i The index of a hexahedron.
     * @return The corresponding hexahedron.
     */
    virtual const Hexahedron getHexahedron(HexaID i) override;

	 /** \brief Get the local hexahedron index (0<i<8) from its 3 binary indices.
     *
     * @param bi array of 3 binary indices (0 or 1 for each component)
     * @return The corresponding local index between 0 and 7.
     */
	virtual unsigned int getLocalIndexFromBinaryIndex(const HexahedronBinaryIndex bi) const;

	 /** \brief Get the binary index (array of 3 binary values) from its local index (0<li<8)
     *
     * @param li local index between 0 and 7 
     * @return its binary index
     */
	virtual HexahedronBinaryIndex getBinaryIndexFromLocalIndex(const unsigned int li) const;

    /** \brief Get the index of a hexahedron from the indices of its vertices.
     *
     * @return The index of the corresponding hexahedron if it exists, -1 otherwise.
     */
    virtual int getHexahedronIndex(PointID v1, PointID v2, PointID v3, PointID v4,
                   PointID v5, PointID v6, PointID v7, PointID v8) override;


    /** \brief Get the 12 edges that form a hexahedron.
     *
     * @param i The index of a hexahedron.
     * @return An EdgesInHexahedron containing the indices of the edges.
     */
    virtual const EdgesInHexahedron& getEdgesInHexahedron(HexaID i) override;


    /** \brief Get the 6 quads that form a hexahedron.
     *
     * @param i The index of a hexahedron.
     * @return A QuadsInHexahedron containing the indices of the quads.
     */
    virtual const QuadsInHexahedron& getQuadsInHexahedron(HexaID i) override;


    /** \brief Get the hexahedra around a vertex.
     *
     * @param i The index of a vertex.
     * @return A HexahedraAroundVertex containing the indices of the hexahedra this vertex belongs to.
     */
    virtual const HexahedraAroundVertex& getHexahedraAroundVertex(PointID i) override;


    /** \brief Get the hexahedra around an edge.
     *
     * @param i The index of an edge.
     * @return A HexahedraAroundEdge containing the indices of the hexahedra this edge belongs to.
     */
    virtual const HexahedraAroundEdge& getHexahedraAroundEdge(EdgeID i) override;


    /** \brief Get the hexahedra around a quad.
     *
     * @param i The index of a quad.
     * @return A HexahedraAroundQuad containing the indices of the hexahedra this quad belongs to.
     */
    virtual const HexahedraAroundQuad& getHexahedraAroundQuad(QuadID i) override;


    /** \brief Get the position of a vertex in a hexahedron from its index.
     *
     * @param t A Hexahedron.
     * @param vertexIndex The index of a vertex.
     * @return The position (between 0 and 7) of this vertex in the Hexahedron if it is present, -1 otherwise.
     */
    int getVertexIndexInHexahedron(const Hexahedron& t, PointID vertexIndex) const override;


    /** \brief Get the position of an edge in a hexahedron from its index.
     *
     * @param t An EdgesInhexahedron.
     * @param edgeIndex The index of an edge.
     * @return The position (between 0 and 11) of this edge in the Hexahedron if it is present, -1 otherwise.
     */
    int getEdgeIndexInHexahedron(const EdgesInHexahedron& t, EdgeID edgeIndex) const override;


    /** \brief Get the position of a quad in a hexahedron from its index.
     *
     * @param t A QuadInHexahedron.
     * @param quadIndex The index of a quad.
     * @return The position (between 0 and 5) of this quad in the Hexahedron if it is present, -1 otherwise.
     */
    int getQuadIndexInHexahedron(const QuadsInHexahedron& t, QuadID quadIndex) const override;


    /** \brief Returns for each index (between 0 and 11) the two vertex local indices that are adjacent to/forming that edge
     *
     */
    virtual Edge getLocalEdgesInHexahedron (const EdgeID i) const override;


    /** \brief Returns for each index (between 0 and 5) the four vertices local indices that are adjacent to/forming that quad
     *
     */
    virtual Quad getLocalQuadsInHexahedron (const QuadID i) const override;

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
    virtual bool checkTopology() const override;


    /// Get information about connexity of the mesh
    /// @{
    /** \brief Checks if the topology has only one connected component
      *
      * @return true if only one connected component
      */
    virtual bool checkConnexity() override;

    /// Returns the number of connected component.
    virtual unsigned int getNumberOfConnectedComponent() override;

    /// Returns the set of element indices connected to an input one (i.e. which can be reached by topological links)
    virtual const VecHexaID getConnectedElement(HexaID elem) override;

    /// Returns the set of element indices adjacent to a given element (i.e. sharing a link)
    virtual const VecHexaID getElementAroundElement(HexaID elem) override;
    /// Returns the set of element indices adjacent to a given list of elements (i.e. sharing a link)
    virtual const VecHexaID getElementAroundElements(VecHexaID elems) override;
    /// @}


    /** \brief Returns the number of hexahedra in this topology.
     *	The difference to getNbHexahedra() is that this method does not generate the hexa array if it does not exist.
     */
    unsigned int getNumberOfHexahedra() const;


    /** \brief Returns the number of topological element of the current topology.
     * This function avoids to know which topological container is in used.
     */
    virtual unsigned int getNumberOfElements() const override;


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

    /** \brief Returns the type of the topology */
	  virtual sofa::core::topology::TopologyObjectType getTopologyType() const override {return sofa::core::topology::HEXAHEDRON;}


protected:

    /** \brief Creates the EdgeSet array.
     *
     * Create the set of edges when needed.
     */
    virtual void createEdgeSetArray() override;


    /** \brief Creates the QuadSet array.
     *
     * Create the array of quads when needed.
     */
    virtual void createQuadSetArray() override;


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


    /// \brief Function creating the data graph linked to d_hexahedron
    virtual void updateTopologyEngineGraph() override;


    /// Use a specific boolean @see m_hexahedronTopologyDirty in order to know if topology Data is dirty or not.
    /// Set/Get function access to this boolean
    void setHexahedronTopologyToDirty() {m_hexahedronTopologyDirty = true;}
    void cleanHexahedronTopologyFromDirty() {m_hexahedronTopologyDirty = false;}
    const bool& isHexahedronTopologyDirty() {return m_hexahedronTopologyDirty;}

public:
	/// force the creation of quads
	Data<bool>  d_createQuadArray;

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


    /// Boolean used to know if the topology Data of this container is dirty
    bool m_hexahedronTopologyDirty;

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
