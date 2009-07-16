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
*                       x                                                      *
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
using core::componentmodel::topology::BaseMeshTopology;

typedef BaseMeshTopology::PointID			PointID;
typedef BaseMeshTopology::EdgeID			EdgeID;
typedef BaseMeshTopology::Edge				Edge;
typedef BaseMeshTopology::SeqEdges			SeqEdges;
typedef BaseMeshTopology::EdgesAroundVertex		EdgesAroundVertex;

/** a class that stores a set of edges  and provides access to the adjacency between points and edges */
class SOFA_COMPONENT_CONTAINER_API EdgeSetTopologyContainer : public PointSetTopologyContainer
{
    friend class EdgeSetTopologyModifier;

public:
    EdgeSetTopologyContainer();

    EdgeSetTopologyContainer(const sofa::helper::vector< Edge > &edges);

    virtual ~EdgeSetTopologyContainer() {}

    /// Procedural creation methods
    /// @{
    virtual void clear();
    virtual void addEdge( int a, int b );
    /// @}

    virtual void init();

    /// BaseMeshTopology API
    /// @{

    virtual const SeqEdges& getEdges()
    {
        return getEdgeArray();
    }

    /** \brief Returns the edge shell of the ith DOF.
    *
    */
    virtual const EdgesAroundVertex& getEdgesAroundVertex(PointID i);


    /** \brief Returns the index of the edge joining vertex v1 and vertex v2; returns -1 if no edge exists
    *
    */
    virtual int getEdgeIndex(PointID v1, PointID v2);

    /// @}

    /** \brief Checks if the topology is coherent
    *
    * Check if the shell arrays are coherent
    */
    virtual bool checkTopology() const;

    /** \brief Returns the number of edges in this topology.
    *  The difference to getNbEdges() is that this method does not generate the edge array if it does not exist.
    */
    unsigned int getNumberOfEdges() const;

    /** \brief Returns the Edge array.
    *
    */
    virtual const sofa::helper::vector<Edge> &getEdgeArray();


    /** \brief Returns the Edge Shell array.
    *
    */
    virtual const sofa::helper::vector< sofa::helper::vector<unsigned int> > &getEdgesAroundVertexArray();


    /** \brief Returns the number of connected components from the graph containing all edges and give, for each vertex, which component it belongs to  (use BOOST GRAPH LIBRAIRY)
    @param components the array containing the optimal vertex permutation according to the Reverse CuthillMckee algorithm
    */
    virtual int getNumberConnectedComponents(sofa::helper::vector<unsigned int>& components);

    /** \brief Returns a non-const edge shell of the ith DOF for subsequent modification
    *
    */
    virtual sofa::helper::vector< unsigned int > &getEdgesAroundVertexForModification(const unsigned int i);

    /** \brief Creates the EdgeSet array.
    *
    * This function must be implemented by derived classes to create a list of edges from a set of triangles or tetrahedra
    */
    virtual void createEdgeSetArray();

    /** \brief Creates the EdgeSetIndex.
    *
    * This function is only called if the EdgeShell member is required.
    * EdgeShell[i] contains the indices of all edges having the ith DOF as
    * one of their ends.
    */
    virtual void createEdgesAroundVertexArray();

    bool hasEdges() const;

    bool hasEdgesAroundVertex() const;

    void clearEdges();

    void clearEdgesAroundVertex();

protected:
    /*** The array that stores the set of edges in the edge set */
    sofa::helper::vector<Edge> m_edge;
    DataPtr< sofa::helper::vector<Edge> > d_edge;

    /** the array that stores the set of edge-vertex shells, ie for each vertex gives the set of adjacent edges */
    sofa::helper::vector< sofa::helper::vector< unsigned int > > m_edgesAroundVertex;

    virtual void loadFromMeshLoader(sofa::component::container::MeshLoader* loader);
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
