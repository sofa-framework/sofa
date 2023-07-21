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

#include <sofa/component/topology/container/dynamic/PointSetTopologyContainer.h>

namespace sofa::component::topology::container::dynamic
{


/** a class that stores a set of edges  and provides access to the adjacency between points and edges */
class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API EdgeSetTopologyContainer : public PointSetTopologyContainer
{
    friend class EdgeSetTopologyModifier;

public:
    SOFA_CLASS(EdgeSetTopologyContainer,PointSetTopologyContainer);

    typedef BaseMeshTopology::PointID               PointID;
    typedef BaseMeshTopology::EdgeID                EdgeID;
    typedef BaseMeshTopology::Edge                  Edge;
    typedef BaseMeshTopology::SeqEdges              SeqEdges;
    typedef BaseMeshTopology::EdgesAroundVertex     EdgesAroundVertex;
    typedef sofa::type::vector<EdgeID>            VecEdgeID;


protected:
    EdgeSetTopologyContainer();

    ~EdgeSetTopologyContainer() override {}
public:
    void init() override;

    void reinit() override;

    /// Procedural creation methods
    /// @{
    void clear() override;
    void addEdge( Index a, Index b ) override;
    /// @}

    /// BaseMeshTopology API
    /// @{

    /** \brief Returns the edge array.
     *
     */
    const SeqEdges& getEdges() override;

    /** \brief Get an Edge from its ID.
     *
     * @param i The ID of the Edge.
     * @return The corresponding Edge.
     */
    const Edge getEdge(EdgeID i) override;


    /** \brief Get the index of the edge joining two vertices.
     *
     * @param v1 The first vertex
     * @param v@ The second vertex
     * @return The index of the Edge if it exists, InvalidID otherwise.
    */
    EdgeID getEdgeIndex(PointID v1, PointID v2) override;


    /** \brief Get the indices of the edges around a vertex.
     *
     * @param i The ID of the vertex.
     * @return An EdgesAroundVertex containing the indices of the edges.
     */
    const EdgesAroundVertex& getEdgesAroundVertex(PointID id) override;

    /// @}



    /// Dynamic Topology API
    /// @{
    /// Method called by component Init method. Will create all the topology neighboorhood buffers.
    void initTopology();

    /** \brief Checks if the topology is coherent
     *
     * Check if the arrays are coherent.
     * @see m_edgesAroundVertex
     * @see m_edge
     * @return bool true if topology is coherent.
     */
    bool checkTopology() const override;


    /** \brief Returns the number of edges in this topology.
     *
     * The difference to getNbEdges() is that this method does not generate the edge array if it does not exist.
     * @return the number of edges.
     */
    Size getNumberOfEdges() const;

    /** \brief Returns the number of topological element of the current topology.
     * This function avoids to know which topological container is in used.
     */
    Size getNumberOfElements() const override;


    /** \brief Returns the number of connected components from the graph containing all edges and give, for each vertex, which component it belongs to  (use BOOST GRAPH LIBRAIRY)
     *
     * @param components the array containing the optimal vertex permutation according to the Reverse CuthillMckee algorithm
     * @return The number of components connected together.
     */
    virtual int getNumberConnectedComponents(sofa::type::vector<EdgeID>& components);


    /** \brief Returns the Edge array.
     *
     */
    virtual const sofa::type::vector<Edge>& getEdgeArray();


    /** \brief Returns the list of Edge indices around each DOF.
     *
     * @return EdgesAroundVertex lists.
     */
    virtual const sofa::type::vector< sofa::type::vector<EdgeID> >& getEdgesAroundVertexArray();


    bool hasEdges() const;

    bool hasEdgesAroundVertex() const;

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
    const VecEdgeID getConnectedElement(EdgeID elem) override;

    /// Returns the set of element indices adjacent to a given element (i.e. sharing a link)
    const VecEdgeID getElementAroundElement(EdgeID elem) override;
    /// Returns the set of element indices adjacent to a given list of elements (i.e. sharing a link)
    const VecEdgeID getElementAroundElements(VecEdgeID elems) override;
    /// @}

    /** \brief Returns the type of the topology */
    sofa::geometry::ElementType getTopologyType() const override {return sofa::geometry::ElementType::EDGE;}

    bool linkTopologyHandlerToData(core::topology::TopologyHandler* topologyHandler, sofa::geometry::ElementType elementType) override;
    
    bool unlinkTopologyHandlerToData(core::topology::TopologyHandler* topologyHandler, sofa::geometry::ElementType elementType) override;

protected:

    /** \brief Creates the EdgeSet array.
     *
     * This function must be implemented by derived classes to create a list of edges from a set of triangles or tetrahedra
     */
    virtual void createEdgeSetArray();


    /** \brief Creates the EdgesAroundVertex array.
    *
    * This function is only called if EdgesAroundVertex member is required.
    * EdgesAroundVertex[i] contains the indices of all edges having the ith DOF as
    * one of their ends.
    */
    virtual void createEdgesAroundVertexArray();


    void clearEdges();

    void clearEdgesAroundVertex();

    /// Use a specific boolean @see m_triangleTopologyDirty in order to know if topology Data is dirty or not.
    /// Set/Get function access to this boolean
    void setEdgeTopologyToDirty();
    void cleanEdgeTopologyFromDirty();
    const bool& isEdgeTopologyDirty() {return m_edgeTopologyDirty;}

protected:

    /** \brief Returns a non-const list of Edge indices around the ith DOF for subsequent modification.
     *
     * @return EdgesAroundVertex lists in non-const.
     * @see getEdgesAroundVertex()
     */
    virtual EdgesAroundVertex &getEdgesAroundVertexForModification(const PointID i);

protected:

    /** the array that stores the set of edge-vertex shells, ie for each vertex gives the set of adjacent edges */
    sofa::type::vector< EdgesAroundVertex > m_edgesAroundVertex;

    /// Boolean used to know if the topology Data of this container is dirty
    bool m_edgeTopologyDirty = false;

public:
    /** The array that stores the set of edges in the edge set */
    Data< sofa::type::vector<Edge> > d_edge; ///< List of edge indices

    Data <bool> m_checkConnexity; ///< It true, will check the connexity of the mesh.


};

} //namespace sofa::component::topology::container::dynamic
