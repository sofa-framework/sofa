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

#include <sofa/component/topology/container/dynamic/EdgeSetTopologyContainer.h>

namespace sofa::component::topology::container::dynamic
{
class QuadSetTopologyModifier;



/** Object that stores a set of quads and provides access
to each quad and its edges and vertices */
class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API QuadSetTopologyContainer : public EdgeSetTopologyContainer
{
    friend class QuadSetTopologyModifier;

public:
    SOFA_CLASS(QuadSetTopologyContainer,EdgeSetTopologyContainer);

    typedef BaseMeshTopology::PointID			PointID;
    typedef BaseMeshTopology::EdgeID			EdgeID;
    typedef BaseMeshTopology::QuadID			QuadID;
    typedef BaseMeshTopology::Edge				Edge;
    typedef BaseMeshTopology::Quad				Quad;
    typedef BaseMeshTopology::SeqQuads			SeqQuads;
    typedef BaseMeshTopology::EdgesInQuad			EdgesInQuad;
    typedef BaseMeshTopology::QuadsAroundVertex		QuadsAroundVertex;
    typedef BaseMeshTopology::QuadsAroundEdge		QuadsAroundEdge;
    typedef sofa::type::vector<QuadID>                  VecQuadID;

protected:
    QuadSetTopologyContainer();

    ~QuadSetTopologyContainer() override {}
public:
    void init() override;


    /// Procedural creation methods
    /// @{
    void clear() override;
    void addQuad(Index a, Index b, Index c, Index d ) override;
    /// @}



    /// BaseMeshTopology API
    /// @{

    /** \brief Returns the quad array.
     *
     */
    const SeqQuads& getQuads() override
    {
        return getQuadArray();
    }


    /** \brief Returns the quad corresponding to the QuadID i.
     *
     * @param ID of a Quad.
     * @return The corresponding Quad.
     */
    const Quad getQuad(QuadID id) override;


    /** Returns the indices of a quad given four vertex indices.
     *
     * @param the four vertex indices.
     * @return the ID of the corresponding quad.
     * @return InvalidID if none
     */
    QuadID getQuadIndex(PointID v1, PointID v2, PointID v3, PointID v4) override;


    /** \brief Returns the set of edges adjacent to a given quad.
     *
     * @param ID of a quad.
     * @return EdgesInQuad list composing the input quad.
    */
    const EdgesInQuad& getEdgesInQuad(QuadID id) override;


    /** \brief Returns the set of quads adjacent to a given vertex.
     *
     * @param ID of a vertex.
     * @return QuadsAroundVertex list around the input vertex.
     */
    const QuadsAroundVertex& getQuadsAroundVertex(PointID id) override;


    /** \brief Returns the set of quads adjacent to a given edge.
     *
     * @param ID of an edge.
     * @return QuadsAroundEdge list around the input edge.
     */
    const QuadsAroundEdge& getQuadsAroundEdge(EdgeID id) override;


    /** \brief Returns the index (either 0, 1, 2, 3) of the vertex whose global index is vertexIndex.
     *
     * @param Ref to a quad.
     * @param Id of a vertex.
     * @return the position of this vertex in the quad (i.e. either 0, 1, 2, 3).
     * @return -1 if none.
     */
    int getVertexIndexInQuad(const Quad &t, PointID vertexIndex) const override;


    /** \brief Returns the index (either 0, 1, 2, 3) of the edge whose global index is edgeIndex.
     *
     * @param Ref to an EdgesInQuad.
     * @param Id of an edge .
     * @return the position of this edge in the quad (i.e. either 0, 1, 2, 3).
     * @return -1 if none.
     */
    int getEdgeIndexInQuad(const EdgesInQuad &t, EdgeID edheIndex) const override;

    /// @}



    /// Dynamic Topology API
    /// @{
    /// Method called by component Init method. Will create all the topology neighboorhood buffers and call @see EdgeSetTopologyContainer::initTopology()
    void initTopology();

    /** \brief Checks if the topology is coherent
    *
    * Check if the shell arrays are coherent
    * @see m_quad
    * @see m_edgesInQuad
    * @see m_quadsAroundVertex
    * @see m_quadsAroundEdge
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
    const VecQuadID getConnectedElement(QuadID elem) override;

    /// Returns the set of element indices adjacent to a given element (i.e. sharing a link)
    const VecQuadID getElementAroundElement(QuadID elem) override;
    /// Returns the set of element indices adjacent to a given list of elements (i.e. sharing a link)
    const VecQuadID getElementAroundElements(VecQuadID elems) override;
    /// @}


    /** \brief Returns the number of quads in this topology.
     * The difference to getNbQuads() is that this method does not generate the quad array if it does not exist.
     */
    Size getNumberOfQuads() const;


    /** \brief Returns the number of topological element of the current topology.
     * This function avoids to know which topological container is in used.
     */
    Size getNumberOfElements() const override;


    /** \brief Returns the Quad array. */
    const sofa::type::vector<Quad> &getQuadArray();


    /** \brief Returns the EdgesInQuadArray array (i.e. provide the 4 edge indices for each quad) */
    const sofa::type::vector< EdgesInQuad > &getEdgesInQuadArray() ;


    /** \brief Returns the QuadsAroundVertex array (i.e. provide the quad indices adjacent to each vertex). */
    const sofa::type::vector< QuadsAroundVertex > &getQuadsAroundVertexArray();


    /** \brief Returns the QuadsAroundEdge array (i.e. provide the quad indices adjacent to each edge). */
    const sofa::type::vector< QuadsAroundEdge > &getQuadsAroundEdgeArray() ;


    bool hasQuads() const;

    bool hasEdgesInQuad() const;

    bool hasQuadsAroundVertex() const;

    bool hasQuadsAroundEdge() const;

    /// @}

    /** \brief Returns the type of the topology */
    sofa::geometry::ElementType getTopologyType() const override {return sofa::geometry::ElementType::QUAD;}

    bool linkTopologyHandlerToData(core::topology::TopologyHandler* topologyHandler, sofa::geometry::ElementType elementType) override;

    bool unlinkTopologyHandlerToData(core::topology::TopologyHandler* topologyHandler, sofa::geometry::ElementType elementType) override;

protected:

    /** \brief Creates the QuadSet array.
     *
     * This function must be implemented by derived classes to create a list of quads from a set of hexahedra for instance.
     */
    virtual void createQuadSetArray();


    /** \brief Creates the EdgeSet array.
     *
     * Create the set of edges when needed.
     */
    void createEdgeSetArray() override;


    /** \brief Creates the array of edge indices for each quad.
     *
     * This function is only called if the EdgesInQuad array is required.
     * m_edgesInQuad[i] contains the 4 indices of the 4 edges composing the ith quad.
     */
    virtual void createEdgesInQuadArray();


    /** \brief Creates the QuadsAroundVertex Array.
     *
     * This function is only called if the QuadsAroundVertex array is required.
     * m_quadsAroundVertex[i] contains the indices of all quads adjacent to the ith vertex.
     */
    virtual void createQuadsAroundVertexArray();


    /** \brief Creates the quadsAroundEdge Array.
     *
     * This function is only called if the QuadsAroundVertex array is required.
     * m_quadsAroundEdge[i] contains the indices of all quads adjacent to the ith edge
     */
    virtual void createQuadsAroundEdgeArray();


    void clearQuads();

    void clearEdgesInQuad();

    void clearQuadsAroundVertex();

    void clearQuadsAroundEdge();


protected:

    /** \brief Returns a non-const list of quad indices around a given DOF for subsequent modification.
     *
     * @return QuadsAroundVertex lists in non-const.
     * @see getQuadsAroundVertex()
     */
    virtual QuadsAroundVertex& getQuadsAroundVertexForModification(const PointID vertexIndex);


    /** \brief Returns a non-const list of quad indices around a given edge for subsequent modification.
     *
     * @return QuadsAroundEdge lists in non-const.
     * @see getQuadsAroundEdge()
     */
    virtual QuadsAroundEdge& getQuadsAroundEdgeForModification(const EdgeID edgeIndex);

    /// Use a specific boolean @see m_quadTopologyDirty in order to know if topology Data is dirty or not.
    /// Set/Get function access to this boolean
    void setQuadTopologyToDirty();
    void cleanQuadTopologyFromDirty();
    const bool& isQuadTopologyDirty() {return m_quadTopologyDirty;}

public:
    /// provides the set of quads.
    Data< sofa::type::vector<Quad> > d_quad;

protected:
    /// provides the 4 edges in each quad.
    sofa::type::vector<EdgesInQuad> m_edgesInQuad;

    /// for each vertex provides the set of quads adjacent to that vertex.
    sofa::type::vector< QuadsAroundVertex > m_quadsAroundVertex;

    /// for each edge provides the set of quads adjacent to that edge.
    sofa::type::vector< QuadsAroundEdge > m_quadsAroundEdge;


    /// Boolean used to know if the topology Data of this container is dirty
    bool m_quadTopologyDirty = false;

};

} //namespace sofa::component::topology::container::dynamic
