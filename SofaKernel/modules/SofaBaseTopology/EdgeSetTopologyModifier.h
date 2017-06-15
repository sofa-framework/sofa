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
#ifndef SOFA_COMPONENT_TOPOLOGY_EDGESETTOPOLOGYMODIFIER_H
#define SOFA_COMPONENT_TOPOLOGY_EDGESETTOPOLOGYMODIFIER_H
#include "config.h"

#include <SofaBaseTopology/PointSetTopologyModifier.h>

namespace sofa
{

namespace component
{

namespace topology
{
class EdgeSetTopologyContainer;


/**
 * A class that can apply basic transformations on a set of edges.
 */
class SOFA_BASE_TOPOLOGY_API EdgeSetTopologyModifier : public PointSetTopologyModifier
{
public:
    SOFA_CLASS(EdgeSetTopologyModifier,PointSetTopologyModifier);

    typedef core::topology::BaseMeshTopology::EdgeID EdgeID;
    typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef core::topology::BaseMeshTopology::SeqEdges SeqEdges;
    typedef core::topology::BaseMeshTopology::EdgesAroundVertex EdgesAroundVertex;
protected:

    EdgeSetTopologyModifier()
        : PointSetTopologyModifier()
    {}

    virtual ~EdgeSetTopologyModifier() {}
public:
    virtual void init();

    /// \brief function to propagate topological change events by parsing the list of topologyEngines linked to this topology.
    virtual void propagateTopologicalEngineChanges();

    /** \brief add a set of edges
    @param edges an array of pair of vertex indices describing the edge to be created
    *
    */
    virtual void addEdges(const sofa::helper::vector< Edge >& edges) ;

    /** \brief add a set of edges
    @param edges an array of pair of vertex indices describing the edge to be created
    @param ancestors for each edge to be created provides an array of edge ancestors (optional)
    @param baryCoefs for each edge provides the barycentric coordinates (sum to 1) associated with each ancestor (optional)
    *
    */
    virtual void addEdges( const sofa::helper::vector< Edge >& edges,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs) ;
    
    /** \brief add a set of edges
    @param edges an array of pair of vertex indices describing the edge to be created
    @param ancestors for each edge to be created provides an array of edge ancestors (optional)
    @param baryCoefs for each edge provides the barycentric coordinates (sum to 1) associated with each ancestor (optional)
    *
    */
    virtual void addEdges( const sofa::helper::vector< Edge >& edges,
        const sofa::helper::vector< core::topology::EdgeAncestorElem >& ancestorElems);

    /** \brief Sends a message to warn that some edges were added in this topology.
    *
    * \sa addEdgesProcess
    */
    virtual void addEdgesWarning(const unsigned int nEdges);

    /** \brief Sends a message to warn that some edges were added in this topology.
    *
    * \sa addEdgesProcess
    */
    virtual void addEdgesWarning(const unsigned int nEdges,
            const sofa::helper::vector< Edge >& edgesList,
            const sofa::helper::vector< unsigned int >& edgesIndexList);

    /** \brief Sends a message to warn that some edges were added in this topology.
    *
    * \sa addEdgesProcess
    */
    virtual void addEdgesWarning(const unsigned int nEdges,
            const sofa::helper::vector< Edge >& edgesList,
            const sofa::helper::vector< unsigned int >& edgesIndexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors);

    /** \brief Sends a message to warn that some edges were added in this topology.
    *
    * \sa addEdgesProcess
    */
    virtual void addEdgesWarning(const unsigned int nEdges,
            const sofa::helper::vector< Edge >& edgesList,
            const sofa::helper::vector< unsigned int >& edgesIndexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs);
    
    /** \brief Sends a message to warn that some edges were added in this topology.
    *
    * \sa addEdgesProcess
    */
    virtual void addEdgesWarning(const unsigned int nEdges,
            const sofa::helper::vector< Edge >& edgesList,
            const sofa::helper::vector< unsigned int >& edgesIndexList,
            const sofa::helper::vector< core::topology::EdgeAncestorElem >& ancestorElems);

    /** \brief Effectively add an edge.
    */
    void addEdgeProcess(Edge e);

    /** \brief Effectively add some edges to this topology.
    *
    * \sa addEdgesWarning
    */
    virtual void addEdgesProcess(const sofa::helper::vector< Edge > &edges);

    /** \brief Sends a message to warn that some edges are about to be deleted.
    *
    * \sa removeEdgesProcess
    */
    // side effect : edges are sorted first
    virtual void removeEdgesWarning(/*const*/ sofa::helper::vector<unsigned int> &edges);

    /** \brief Effectively Remove a subset of edges. Eventually remove isolated vertices
    *
    * Elements corresponding to these edges are removed form the mechanical object's state vectors.
    *
    * Important : some structures might need to be warned BEFORE the edges are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * \sa removeEdgesWarning
    *
    * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
    *
    * @param removeIsolatedItems if true isolated vertices are also removed
    */
    virtual void removeEdgesProcess(const sofa::helper::vector<unsigned int> &indices, const bool removeIsolatedItems = false);

    /** \brief Swap the edges.
    *
    */
    virtual void swapEdgesProcess(const sofa::helper::vector< sofa::helper::vector< unsigned int > >& edgesPairs);

    /** \brief Fuse the edges.
    *
    * @param removeIsolatedItems if true isolated vertices are also removed
    */
    virtual void fuseEdgesProcess(const sofa::helper::vector< sofa::helper::vector< unsigned int > >& edgesPairs, const bool removeIsolatedPoints = true);

    /** \brief Split the edges.
    *
    * @param removeIsolatedItems if true isolated vertices are also removed
    */
    virtual void splitEdgesProcess(/*const*/ sofa::helper::vector<unsigned int> &indices,
            const bool removeIsolatedPoints = true);

    /** \brief Split the edges.
    *
    * @param removeIsolatedItems if true isolated vertices are also removed
    */
    virtual void splitEdgesProcess(/*const*/ sofa::helper::vector<unsigned int> &indices,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
            const bool removeIsolatedPoints = true);

    /** \brief Add some points to this topology.
    *
    * \sa addPointsWarning
    */
    virtual void addPointsProcess(const unsigned int nPoints);

    /** \brief Remove a subset of points
    *
    * these points are removed form the mechanical object's state vectors.
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removePointsWarning before calling removePointsProcess.
    * \sa removePointsWarning
    * Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    */
    virtual void removePointsProcess(const sofa::helper::vector<unsigned int> &indices,
            const bool removeDOF = true);

    /** \brief Reorder this topology.
    *
    * Important : the points are actually renumbered in the mechanical object's state vectors iff (renumberDOF == true)
    * \see MechanicalObject::renumberValues
    */
    virtual void renumberPointsProcess( const sofa::helper::vector<unsigned int> &index,
            const sofa::helper::vector<unsigned int> &/*inv_index*/,
            const bool renumberDOF = true);

    /** \brief Remove a set of edges
    @param edges an array of edge indices to be removed (note that the array is not const since it needs to be sorted)
    *
    */
    virtual void removeEdges(const sofa::helper::vector<unsigned int> &edgeIds,
            const bool removeIsolatedPoints = true, const bool resetTopoChange = true);

    /** \brief Generic method to remove a list of items.
    */
    virtual void removeItems(const sofa::helper::vector<unsigned int> &items);

    /** \brief Generic method for points renumbering
    */
    virtual void renumberPoints( const sofa::helper::vector<unsigned int> & index,
            const sofa::helper::vector<unsigned int> & inv_index);

    /** \brief Swap a list of pair edges, replacing each edge pair ((p11, p12), (p21, p22)) by the edge pair ((p11, p21), (p12, p22))
    *
    */
    virtual void swapEdges(const sofa::helper::vector< sofa::helper::vector< unsigned int > >& edgesPairs);

    /** \brief Fuse a list of pair edges, replacing each edge pair ((p11, p12), (p21, p22)) by one edge (p11, p22)
    *
    * @param removeIsolatedPoints if true isolated vertices are also removed
    */
    virtual void fuseEdges(const sofa::helper::vector< sofa::helper::vector< unsigned int > >& edgesPairs, const bool removeIsolatedPoints = true);

    /** \brief Split an array of edges, replacing each edge (p1, p2) by two edges (p1, p3) and (p3, p2) where p3 is the new vertex
    * On each edge, a vertex is created based on its barycentric coordinates
    *
    * @param removeIsolatedPoints if true isolated vertices are also removed
    */
    virtual void splitEdges( sofa::helper::vector<unsigned int> &indices,
            const bool removeIsolatedPoints = true);

    /** \brief Split an array of edges, replacing each edge (p1, p2) by two edges (p1, p3) and (p3, p2) where p3 is the new vertex
    * On each edge, a vertex is created based on its barycentric coordinates
    *
    * @param removeIsolatedPoints if true isolated vertices are also removed
    */
    virtual void splitEdges( sofa::helper::vector<unsigned int> &indices,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
            const bool removeIsolatedPoints = true);

    /** \brief Gives the optimal vertex permutation according to the Reverse CuthillMckee algorithm (use BOOST GRAPH LIBRAIRY)
    */
    virtual void resortCuthillMckee(sofa::helper::vector<int>& inverse_permutation);


    /** \brief Move input points indices to input new coords.
     * Also propagate event and update edgesAroundVertex for data handling.
     *
     * @param id The list of indices to move
     * @param ancestors The list of ancestors to define relative new position
     * @param coefs The barycoef to locate new coord relatively to ancestors.
     * @moveDOF bool allowing the move (default true)
     */
    virtual void movePointsProcess (const sofa::helper::vector <unsigned int>& id,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs,
            const bool moveDOF = true);



    /** \brief Given an element indice, it will remove all the connected component in which this element belongs to.
    *  Warning: if there is only one connected component in the model. All the object will be removed.
    *
    * @param elemID The ID of the input element.
    * @return false if something goes wrong during the process.
    */
    virtual bool removeConnectedComponents(unsigned int elemID);

    /** \brief Given an element indice, it will remove all elements directly connected to the input one.
    *
    * @param elemID The ID of the input element.
    * @return false if something goes wrong during the process.
    */
    virtual bool removeConnectedElements(unsigned int elemID);

    /** \brief If several connected components are detected, it will keep only the biggest one and remove all the rest.
    * Warning: if two connected components have the same number of element and are the biggest. It will keep the first one.
    *
    * @return false if something goes wrong during the process.
    */
    virtual bool removeIsolatedElements();

    /** \brief If several connected components are detected, it will remove all connected component with less than a given number of elements.
    *
    * @param scaleElem: threshold number size under which connected component will be removed.
    * @return false if something goes wrong during the process.
    */
    virtual bool removeIsolatedElements(unsigned int scaleElem);

private:
    EdgeSetTopologyContainer* 	m_container;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
