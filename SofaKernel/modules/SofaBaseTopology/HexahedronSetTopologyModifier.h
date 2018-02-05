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
#ifndef SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGYMODIFIER_H
#define SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGYMODIFIER_H
#include "config.h"

#include <SofaBaseTopology/QuadSetTopologyModifier.h>

namespace sofa
{
namespace component
{
namespace topology
{
class HexahedronSetTopologyContainer;


/**
* A class that modifies the topology by adding and removing hexahedra
*/
class SOFA_BASE_TOPOLOGY_API HexahedronSetTopologyModifier : public QuadSetTopologyModifier
{
public:
    SOFA_CLASS(HexahedronSetTopologyModifier,QuadSetTopologyModifier);


    typedef core::topology::BaseMeshTopology::HexaID HexaID;
    typedef core::topology::BaseMeshTopology::Hexa Hexa;
    typedef core::topology::BaseMeshTopology::SeqHexahedra SeqHexahedra;
    typedef core::topology::BaseMeshTopology::HexahedraAroundVertex HexahedraAroundVertex;
    typedef core::topology::BaseMeshTopology::HexahedraAroundEdge HexahedraAroundEdge;
    typedef core::topology::BaseMeshTopology::HexahedraAroundQuad HexahedraAroundQuad;
    typedef core::topology::BaseMeshTopology::EdgesInHexahedron EdgesInHexahedron;
    typedef core::topology::BaseMeshTopology::QuadsInHexahedron QuadsInHexahedron;

    typedef Hexa Hexahedron;

protected:
    HexahedronSetTopologyModifier()
        : QuadSetTopologyModifier()
        , removeIsolated( initData(&removeIsolated,true, "removeIsolated", "remove Isolated dof") )
    { }

    virtual ~HexahedronSetTopologyModifier() override {}
public:
    virtual void init() override;

    Data< bool > removeIsolated; ///< Controlled DOF index.

    /// \brief function to propagate topological change events by parsing the list of topologyEngines linked to this topology.
    virtual void propagateTopologicalEngineChanges() override;


    /** \brief add a set of hexahedra
    @param hexahedra an array of vertex indices describing the hexahedra to be created
    */
    virtual void addHexahedra(const sofa::helper::vector< Hexahedron > &hexahedra);

    /** \brief add a set of hexahedra
    @param quads an array of vertex indices describing the hexahedra to be created
    @param ancestors for each hexahedron to be created provides an array of hexahedron ancestors (optional)
    @param baryCoefs for each hexahedron provides the barycentric coordinates (sum to 1) associated with each ancestor (optional)
    *
    */
    virtual void addHexahedra(const sofa::helper::vector< Hexahedron > &hexahedra,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs) ;

    /** \brief Sends a message to warn that some hexahedra were added in this topology.
    *
    * \sa addHexahedraProcess
    */
    void addHexahedraWarning(const unsigned int nHexahedra,
            const sofa::helper::vector< Hexahedron >& hexahedraList,
            const sofa::helper::vector< unsigned int >& hexahedraIndexList);

    /** \brief Sends a message to warn that some hexahedra were added in this topology.
    *
    * \sa addHexahedraProcess
    */
    void addHexahedraWarning(const unsigned int nHexahedra,
            const sofa::helper::vector< Hexahedron >& hexahedraList,
            const sofa::helper::vector< unsigned int >& hexahedraIndexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs);

    /** \brief Add a hexahedron.
    */
    void addHexahedronProcess(Hexahedron e);

    /** \brief Actually Add some hexahedra to this topology.
    *
    * \sa addHexahedraWarning
    */
    virtual void addHexahedraProcess(const sofa::helper::vector< Hexahedron > &hexahedra);

    /** \brief Sends a message to warn that some hexahedra are about to be deleted.
    *
    * \sa removeHexahedraProcess
    *
    * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
    */
    virtual void removeHexahedraWarning( sofa::helper::vector<unsigned int> &hexahedra);

    /** \brief Remove a subset of hexahedra
    *
    * Elements corresponding to these points are removed form the mechanical object's state vectors.
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * \sa removeHexahedraWarning
    * @param removeIsolatedItems if true remove isolated quads, edges and vertices
    */
    virtual void removeHexahedraProcess(const sofa::helper::vector<unsigned int>&indices,
            const bool removeIsolatedItems = false);

    /** \brief Actually Add some quads to this topology.
    *
    * \sa addQuadsWarning
    */
    virtual void addQuadsProcess(const sofa::helper::vector< Quad > &quads) override;

    /** \brief Remove a subset of quads
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * @param removeIsolatedEdges if true isolated edges are also removed
    * @param removeIsolatedPoints if true isolated vertices are also removed
    */
    virtual void removeQuadsProcess(const sofa::helper::vector<unsigned int> &indices,
            const bool removeIsolatedEdges = false,
            const bool removeIsolatedPoints = false) override;

    /** \brief Add some edges to this topology.
    *
    * \sa addEdgesWarning
    */
    virtual void addEdgesProcess(const sofa::helper::vector< Edge > &edges) override;

    /** \brief Remove a subset of edges
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * \sa removeEdgesWarning
    *
    * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
    * @param removeIsolatedItems if true remove isolated vertices
    */
    virtual void removeEdgesProcess(const sofa::helper::vector<unsigned int> &indices,
            const bool removeIsolatedItems = false) override;

    /** \brief Add some points to this topology.
    *
    * \sa addPointsWarning
    */
    virtual void addPointsProcess(const unsigned int nPoints) override;

    /** \brief Remove a subset of points
    *
    * Elements corresponding to these points are removed form the mechanical object's state vectors.
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removePointsWarning before calling removePointsProcess.
    * \sa removePointsWarning
    * Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    */
    virtual void removePointsProcess(const sofa::helper::vector<unsigned int> &indices, const bool removeDOF = true) override;

    /** \brief Reorder this topology.
    *
    * Important : the points are actually renumbered in the mechanical object's state vectors iff (renumberDOF == true)
    * \see MechanicalObject::renumberValues
    */
    virtual void renumberPointsProcess( const sofa::helper::vector<unsigned int> &index,
            const sofa::helper::vector<unsigned int>& inv_index,
            const bool renumberDOF = true) override;

    /** \brief Remove a set  of hexahedra
    @param hexahedra an array of hexahedron indices to be removed (note that the array is not const since it needs to be sorted)
    *
    */
    virtual void removeHexahedra(const sofa::helper::vector<unsigned int> &hexahedraIds);

    /** \brief Generic method to remove a list of items.
    */
    virtual void removeItems(const sofa::helper::vector<unsigned int> &items) override;

    /** \brief Generic method for points renumbering
    */
    virtual void renumberPoints( const sofa::helper::vector<unsigned int>& index,
            const sofa::helper::vector<unsigned int>& inv_index) override;


private:
    HexahedronSetTopologyContainer* 	m_container;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
