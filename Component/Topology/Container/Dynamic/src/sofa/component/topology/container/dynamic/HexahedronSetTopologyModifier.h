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

#include <sofa/component/topology/container/dynamic/QuadSetTopologyModifier.h>

namespace sofa::component::topology::container::dynamic
{
class HexahedronSetTopologyContainer;

template <class DataTypes>
class HexahedronSetGeometryAlgorithms;

/**
* A class that modifies the topology by adding and removing hexahedra
*/
class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API HexahedronSetTopologyModifier : public QuadSetTopologyModifier
{
public:
    SOFA_CLASS(HexahedronSetTopologyModifier,QuadSetTopologyModifier);

    template <class DataTypes>
    friend class HexahedronSetGeometryAlgorithms;

    typedef core::topology::BaseMeshTopology::HexaID HexaID;
    typedef core::topology::BaseMeshTopology::HexahedronID HexahedronID;
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

    ~HexahedronSetTopologyModifier() override {}
public:
    void init() override;

    Data< bool > removeIsolated; ///< Controlled DOF index.

    /** \brief add a set of hexahedra
    @param hexahedra an array of vertex indices describing the hexahedra to be created
    */
    virtual void addHexahedra(const sofa::type::vector< Hexahedron > &hexahedra);

    /** \brief add a set of hexahedra
    @param quads an array of vertex indices describing the hexahedra to be created
    @param ancestors for each hexahedron to be created provides an array of hexahedron ancestors (optional)
    @param baryCoefs for each hexahedron provides the barycentric coordinates (sum to 1) associated with each ancestor (optional)
    *
    */
    virtual void addHexahedra(const sofa::type::vector< Hexahedron > &hexahedra,
            const sofa::type::vector< sofa::type::vector< HexahedronID > > & ancestors,
            const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs) ;

    /** \brief Add a hexahedron.
    */
    void addHexahedronProcess(Hexahedron e);

    /** \brief Remove a set  of hexahedra
    @param hexahedra an array of hexahedron indices to be removed (note that the array is not const since it needs to be sorted)
    *
    */
    virtual void removeHexahedra(const sofa::type::vector<HexahedronID> &hexahedraIds);

    /** \brief Generic method to remove a list of items.
    */
    void removeItems(const sofa::type::vector<HexahedronID> &items) override;

protected:
    /** \brief Sends a message to warn that some hexahedra were added in this topology.
    *
    * \sa addHexahedraProcess
    */
    void addHexahedraWarning(const size_t nHexahedra,
        const sofa::type::vector< Hexahedron >& hexahedraList,
        const sofa::type::vector< HexahedronID >& hexahedraIndexList);

    /** \brief Sends a message to warn that some hexahedra were added in this topology.
    *
    * \sa addHexahedraProcess
    */
    void addHexahedraWarning(const size_t nHexahedra,
        const sofa::type::vector< Hexahedron >& hexahedraList,
        const sofa::type::vector< HexahedronID >& hexahedraIndexList,
        const sofa::type::vector< sofa::type::vector< HexahedronID > >& ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs);

    /** \brief Actually Add some hexahedra to this topology.
    *
    * \sa addHexahedraWarning
    */
    virtual void addHexahedraProcess(const sofa::type::vector< Hexahedron >& hexahedra);

    /** \brief Sends a message to warn that some hexahedra are about to be deleted.
    *
    * \sa removeHexahedraProcess
    *
    * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
    */
    virtual void removeHexahedraWarning(sofa::type::vector<HexahedronID>& hexahedra);

    /** \brief Remove a subset of hexahedra
    *
    * Elements corresponding to these points are removed form the mechanical object's state vectors.
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * \sa removeHexahedraWarning
    * @param removeIsolatedItems if true remove isolated quads, edges and vertices
    */
    virtual void removeHexahedraProcess(const sofa::type::vector<HexahedronID>& indices,
        const bool removeIsolatedItems = false);

    /** \brief Actually Add some quads to this topology.
    *
    * \sa addQuadsWarning
    */
    void addQuadsProcess(const sofa::type::vector< Quad >& quads) override;

    /** \brief Remove a subset of quads
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * @param removeIsolatedEdges if true isolated edges are also removed
    * @param removeIsolatedPoints if true isolated vertices are also removed
    */
    void removeQuadsProcess(const sofa::type::vector<QuadID>& indices,
        const bool removeIsolatedEdges = false,
        const bool removeIsolatedPoints = false) override;


    /** \brief Add some edges to this topology.
    *
    * \sa addEdgesWarning
    */
    void addEdgesProcess(const sofa::type::vector< Edge >& edges) override;

    /** \brief Remove a subset of edges
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * \sa removeEdgesWarning
    *
    * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
    * @param removeIsolatedItems if true remove isolated vertices
    */
    void removeEdgesProcess(const sofa::type::vector<EdgeID>& indices,
        const bool removeIsolatedItems = false) override;


    /** \brief Add some points to this topology.
    *
    * \sa addPointsWarning
    */
    void addPointsProcess(const sofa::Size nPoints) override;

    /** \brief Remove a subset of points
    *
    * Elements corresponding to these points are removed form the mechanical object's state vectors.
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removePointsWarning before calling removePointsProcess.
    * \sa removePointsWarning
    * Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    */
    void removePointsProcess(const sofa::type::vector<PointID>& indices, const bool removeDOF = true) override;

    /** \brief Reorder this topology.
    *
    * Important : the points are actually renumbered in the mechanical object's state vectors iff (renumberDOF == true)
    * \see MechanicalObject::renumberValues
    */
    void renumberPointsProcess(const sofa::type::vector<PointID>& index,
        const sofa::type::vector<PointID>& inv_index,
        const bool renumberDOF = true) override;


    /// \brief function to propagate topological change events by parsing the list of TopologyHandlers linked to this topology.
    void propagateTopologicalEngineChanges() override;

private:
    HexahedronSetTopologyContainer* 	m_container;
};

} //namespace sofa::component::topology::container::dynamic
