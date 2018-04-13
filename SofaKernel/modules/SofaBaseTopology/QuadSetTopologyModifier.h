/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_QUADSETTOPOLOGYMODIFIER_H
#define SOFA_COMPONENT_TOPOLOGY_QUADSETTOPOLOGYMODIFIER_H
#include "config.h"

#include <SofaBaseTopology/EdgeSetTopologyModifier.h>

namespace sofa
{

namespace component
{

namespace topology
{
class QuadSetTopologyContainer;


/**
* A class that modifies the topology by adding and removing quads
*/
class SOFA_BASE_TOPOLOGY_API QuadSetTopologyModifier : public EdgeSetTopologyModifier
{
public:
    SOFA_CLASS(QuadSetTopologyModifier,EdgeSetTopologyModifier);

    typedef core::topology::BaseMeshTopology::QuadID QuadID;
    typedef core::topology::BaseMeshTopology::Quad Quad;
    typedef core::topology::BaseMeshTopology::SeqQuads SeqQuads;
    typedef core::topology::BaseMeshTopology::QuadsAroundVertex QuadsAroundVertex;
    typedef core::topology::BaseMeshTopology::QuadsAroundEdge QuadsAroundEdge;
    typedef core::topology::BaseMeshTopology::EdgesInQuad EdgesInQuad;
protected:
    QuadSetTopologyModifier()
        : EdgeSetTopologyModifier()
    { }

    virtual ~QuadSetTopologyModifier() override {}
public:
    virtual void init() override;

    /// \brief function to propagate topological change events by parsing the list of topologyEngines linked to this topology.
    virtual void propagateTopologicalEngineChanges() override;

    /** \brief add a set of quads
    @param quads an array of vertex indices describing the quads to be created
    */
    virtual void addQuads(const sofa::helper::vector< Quad > &quads);

    /** \brief add a set of quads
    @param quads an array of vertex indices describing the quads to be created
    @param ancestors for each quad to be created provides an array of quad ancestors (optional)
    @param baryCoefs for each quad provides the barycentric coordinates (sum to 1) associated with each ancestor (optional)
    *
    */
    virtual void addQuads(const sofa::helper::vector< Quad > &quads,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs) ;


    /** \brief Sends a message to warn that some quads were added in this topology.
    *
    * \sa addQuadsProcess
    */
    void addQuadsWarning(const unsigned int nQuads,
            const sofa::helper::vector< Quad >& quadsList,
            const sofa::helper::vector< unsigned int >& quadsIndexList);

    /** \brief Sends a message to warn that some quads were added in this topology.
    *
    * \sa addQuadsProcess
    */
    void addQuadsWarning(const unsigned int nQuads,
            const sofa::helper::vector< Quad >& quadsList,
            const sofa::helper::vector< unsigned int >& quadsIndexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs);

    /** \brief Effectively Add a quad.
    */
    void addQuadProcess(Quad e);

    /** \brief Effectively Add some quads to this topology.
    *
    	* \sa addQuadsWarning
    	*/
    virtual void addQuadsProcess(const sofa::helper::vector< Quad > &quads);

    /** \brief Sends a message to warn that some quads are about to be deleted.
    *
    * \sa removeQuadsProcess
    *
    * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
    */
    virtual void removeQuadsWarning( sofa::helper::vector<unsigned int> &quads);

    /** \brief Remove a subset of  quads. Eventually remove isolated edges and vertices
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * \sa removeQuadsWarning
    *
    * @param removeIsolatedEdges if true isolated edges are also removed
    * @param removeIsolatedPoints if true isolated vertices are also removed
    */
    virtual void removeQuadsProcess( const sofa::helper::vector<unsigned int> &indices,
            const bool removeIsolatedEdges=false,
            const bool removeIsolatedPoints=false);

    /** \brief Add some edges to this topology.
    *
    * \sa addEdgesWarning
    */
    void addEdgesProcess(const sofa::helper::vector< Edge > &edges) override;

    /** \brief Remove a subset of edges
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * \sa removeEdgesWarning
    *
    * @param removeIsolatedItems if true isolated vertices are also removed
    * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
    */
    virtual void removeEdgesProcess( const sofa::helper::vector<unsigned int> &indices,
            const bool removeIsolatedItems=false) override;

    /** \brief Add some points to this topology.
    *
    * \sa addPointsWarning
    */
    virtual void addPointsProcess(const unsigned int nPoints) override;

    /** \brief Remove a subset of points
    *
    * Elements corresponding to these points are removed from the mechanical object's state vectors.
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removePointsWarning before calling removePointsProcess.
    * \sa removePointsWarning
    * Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    */
    virtual void removePointsProcess(const sofa::helper::vector<unsigned int> &indices,
            const bool removeDOF = true) override;

    /** \brief Reorder this topology.
    *
    * Important : the points are actually renumbered in the mechanical object's state vectors iff (renumberDOF == true)
    * \see MechanicalObject::renumberValues
    */
    virtual void renumberPointsProcess( const sofa::helper::vector<unsigned int>& index,
            const sofa::helper::vector<unsigned int>& inv_index,
            const bool renumberDOF = true) override;

    /** \brief Remove a set  of quads
    @param quads an array of quad indices to be removed (note that the array is not const since it needs to be sorted)
    *
    @param removeIsolatedEdges if true isolated edges are also removed
    @param removeIsolatedPoints if true isolated vertices are also removed
    *
    */
    virtual void removeQuads(const sofa::helper::vector<unsigned int> &quadIds,
            const bool removeIsolatedEdges,
            const bool removeIsolatedPoints);

    /** \brief Generic method to remove a list of items.
    */
    virtual void removeItems(const sofa::helper::vector< unsigned int >& items) override;

    /** \brief Generic method for points renumbering
    */
    virtual void renumberPoints( const sofa::helper::vector<unsigned int>& index,
            const sofa::helper::vector<unsigned int>& inv_index) override;


private:
    QuadSetTopologyContainer* 	m_container;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
