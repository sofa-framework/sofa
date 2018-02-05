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
#ifndef SOFA_COMPONENT_TOPOLOGY_EDGESETTOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_EDGESETTOPOLOGYCONTAINER_H
#include "config.h"

#include <SofaBaseTopology/PointSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace topology
{


/** a class that stores a set of edges  and provides access to the adjacency between points and edges */
class SOFA_BASE_TOPOLOGY_API EdgeSetTopologyContainer : public PointSetTopologyContainer
{
    friend class EdgeSetTopologyModifier;

public:
    SOFA_CLASS(EdgeSetTopologyContainer,PointSetTopologyContainer);

    typedef BaseMeshTopology::PointID		   	PointID;
    typedef BaseMeshTopology::EdgeID			      EdgeID;
    typedef BaseMeshTopology::Edge				   Edge;
    typedef BaseMeshTopology::SeqEdges			   SeqEdges;
    typedef BaseMeshTopology::EdgesAroundVertex	EdgesAroundVertex;
    typedef sofa::helper::vector<EdgeID>         VecEdgeID;


protected:
    EdgeSetTopologyContainer();

    virtual ~EdgeSetTopologyContainer() {}
public:

    virtual void init() override;

    virtual void reinit() override;



    /// Procedural creation methods
    /// @{
    virtual void clear() override;
    virtual void addEdge( int a, int b ) override;
    /// @}



    /// BaseMeshTopology API
    /// @{

    /** \brief Returns the edge array.
     *
     */
    virtual const SeqEdges& getEdges() override
    {
        return getEdgeArray();
    }

    /** \brief Get an Edge from its ID.
     *
     * @param i The ID of the Edge.
     * @return The corresponding Edge.
     */
    virtual const Edge getEdge(EdgeID i) override;


    /** \brief Get the index of the edge joining two vertices.
     *
     * @param v1 The first vertex
     * @param v@ The second vertex
     * @return The index of the Edge if it exists, -1 otherwise.
    */
    virtual int getEdgeIndex(PointID v1, PointID v2) override;


    /** \brief Get the indices of the edges around a vertex.
     *
     * @param i The ID of the vertex.
     * @return An EdgesAroundVertex containing the indices of the edges.
     */
    virtual const EdgesAroundVertex& getEdgesAroundVertex(PointID i) override;

    /// @}



    /// Dynamic Topology API
    /// @{

    /** \brief Checks if the topology is coherent
     *
     * Check if the arrays are coherent.
     * @see m_edgesAroundVertex
     * @see m_edge
     * @return bool true if topology is coherent.
     */
    virtual bool checkTopology() const override;


    /** \brief Returns the number of edges in this topology.
     *
     * The difference to getNbEdges() is that this method does not generate the edge array if it does not exist.
     * @return the number of edges.
     */
    unsigned int getNumberOfEdges() const;

    /** \brief Returns the number of topological element of the current topology.
     * This function avoids to know which topological container is in used.
     */
    virtual unsigned int getNumberOfElements() const override;


    /** \brief Returns the number of connected components from the graph containing all edges and give, for each vertex, which component it belongs to  (use BOOST GRAPH LIBRAIRY)
     *
     * @param components the array containing the optimal vertex permutation according to the Reverse CuthillMckee algorithm
     * @return The number of components connected together.
     */
    virtual int getNumberConnectedComponents(sofa::helper::vector<unsigned int>& components);


    /** \brief Returns the Edge array.
     *
     */
    virtual const sofa::helper::vector<Edge>& getEdgeArray();

    /** \brief Returns a reference to the Data of edges array container. */
    Data< sofa::helper::vector<Edge> >& getEdgeDataArray() {return d_edge;}


    /** \brief Returns the list of Edge indices around each DOF.
     *
     * @return EdgesAroundVertex lists.
     */
    virtual const sofa::helper::vector< sofa::helper::vector<EdgeID> >& getEdgesAroundVertexArray();


    bool hasEdges() const;

    bool hasEdgesAroundVertex() const;

    /// @}


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
    virtual const VecEdgeID getConnectedElement(EdgeID elem) override;

    /// Returns the set of element indices adjacent to a given element (i.e. sharing a link)
    virtual const VecEdgeID getElementAroundElement(EdgeID elem) override;
    /// Returns the set of element indices adjacent to a given list of elements (i.e. sharing a link)
    virtual const VecEdgeID getElementAroundElements(VecEdgeID elems) override;
    /// @}

	  /** \brief Returns the type of the topology */
	  virtual sofa::core::topology::TopologyObjectType getTopologyType() const override {return sofa::core::topology::EDGE;}
    

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


    /// \brief Function creating the data graph linked to d_triangle
    virtual void updateTopologyEngineGraph() override;


    /// Use a specific boolean @see m_triangleTopologyDirty in order to know if topology Data is dirty or not.
    /// Set/Get function access to this boolean
    void setEdgeTopologyToDirty() {m_edgeTopologyDirty = true;}
    void cleanEdgeTopologyFromDirty() {m_edgeTopologyDirty = false;}
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
    sofa::helper::vector< EdgesAroundVertex > m_edgesAroundVertex;


    /// Boolean used to know if the topology Data of this container is dirty
    bool m_edgeTopologyDirty;

    /// List of engines related to this specific container
    std::list<sofa::core::topology::TopologyEngine *> m_enginesList;

    /// \brief variables used to display the graph of Data/DataEngines linked to this Data array.
    sofa::helper::vector < sofa::helper::vector <std::string> > m_dataGraph;
    sofa::helper::vector < sofa::helper::vector <std::string> > m_enginesGraph;

public:
    /** The array that stores the set of edges in the edge set */
    Data< sofa::helper::vector<Edge> > d_edge;

    Data <bool> m_checkConnexity;


};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
