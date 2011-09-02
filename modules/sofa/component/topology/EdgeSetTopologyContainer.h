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
#ifndef SOFA_COMPONENT_TOPOLOGY_EDGESETTOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_EDGESETTOPOLOGYCONTAINER_H

#include <sofa/component/topology/PointSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace topology
{
using core::topology::BaseMeshTopology;

typedef BaseMeshTopology::PointID		   	PointID;
typedef BaseMeshTopology::EdgeID			      EdgeID;
typedef BaseMeshTopology::Edge				   Edge;
typedef BaseMeshTopology::SeqEdges			   SeqEdges;
typedef BaseMeshTopology::EdgesAroundVertex	EdgesAroundVertex;
typedef sofa::helper::vector<EdgeID>         VecEdgeID;

/** a class that stores a set of edges  and provides access to the adjacency between points and edges */
class SOFA_COMPONENT_CONTAINER_API EdgeSetTopologyContainer : public PointSetTopologyContainer
{
    friend class EdgeSetTopologyModifier;

public:
    SOFA_CLASS(EdgeSetTopologyContainer,PointSetTopologyContainer);

    EdgeSetTopologyContainer();

    virtual ~EdgeSetTopologyContainer() {}


    virtual void init();

    virtual void reinit();



    /// Procedural creation methods
    /// @{
    virtual void clear();
    virtual void addEdge( int a, int b );
    /// @}



    /// BaseMeshTopology API
    /// @{

    /** \brief Returns the edge array.
     *
     */
    virtual const SeqEdges& getEdges()
    {
        return getEdgeArray();
    }

    /** \brief Returns the edge corresponding to the EdgeID i.
     *
     * @param ID of an Edge.
     * @return The corresponding Edge.
     */
    virtual const Edge getEdge(EdgeID i);


    /** \brief Returns the index of the edge joining vertex v1 and vertex v2.
     *
     * @param Two vertex IDs v1 and v2.
     * @return The Edge index formed by v1 and v2.
     * @return -1 if no edge exists.
    */
    virtual int getEdgeIndex(PointID v1, PointID v2);


    /** \brief Returns the list of Edge indices around the ith DOF.
     *
     * @param ID of a vertex.
     * @return EdgesAroundVertex list around ith DOF.
     */
    virtual const EdgesAroundVertex& getEdgesAroundVertex(PointID i);

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
    virtual bool checkTopology() const;


    /** \brief Returns the number of edges in this topology.
     *
     * The difference to getNbEdges() is that this method does not generate the edge array if it does not exist.
     * @return the number of edges.
     */
    unsigned int getNumberOfEdges() const;

    /** \brief Returns the number of topological element of the current topology.
     * This function avoids to know which topological container is in used.
     */
    virtual unsigned int getNumberOfElements() const;


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
    virtual bool checkConnexity();

    /// Returns the number of connected component.
    virtual unsigned int getNumberOfConnectedComponent();

    /// Returns the set of element indices connected to an input one (i.e. which can be reached by topological links)
    virtual const VecEdgeID getConnectedElement(EdgeID elem);

    /// Returns the set of element indices adjacent to a given element (i.e. sharing a link)
    virtual const VecEdgeID getElementAroundElement(EdgeID elem);
    /// Returns the set of element indices adjacent to a given list of elements (i.e. sharing a link)
    virtual const VecEdgeID getElementAroundElements(VecEdgeID elems);
    /// @}



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

#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
    virtual void updateTopologyEngineGraph();

    /// graph map
    sofa::helper::vector < sofa::helper::vector <std::string> > m_dataGraph;
    sofa::helper::vector < sofa::helper::vector <std::string> > m_enginesGraph;

    sofa::helper::list <sofa::core::topology::TopologyEngine *> m_enginesList;

    //bool m_topologyDirty;
#endif

protected:

    /** \brief Returns a non-const list of Edge indices around the ith DOF for subsequent modification.
     *
     * @return EdgesAroundVertex lists in non-const.
     * @see getEdgesAroundVertex()
     */
    virtual EdgesAroundVertex &getEdgesAroundVertexForModification(const PointID i);

protected:

    /** The array that stores the set of edges in the edge set */
    Data< sofa::helper::vector<Edge> > d_edge;

    /** the array that stores the set of edge-vertex shells, ie for each vertex gives the set of adjacent edges */
    sofa::helper::vector< EdgesAroundVertex > m_edgesAroundVertex;

public:

    Data <bool> m_checkConnexity;


};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
