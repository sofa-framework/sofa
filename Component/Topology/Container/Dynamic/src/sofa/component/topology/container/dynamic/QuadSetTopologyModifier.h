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

#include <sofa/component/topology/container/dynamic/EdgeSetTopologyModifier.h>

namespace sofa::component::topology::container::dynamic
{
class QuadSetTopologyContainer;

template <class DataTypes>
class QuadSetGeometryAlgorithms;

/**
* A class that modifies the topology by adding and removing quads
*/
class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API QuadSetTopologyModifier : public EdgeSetTopologyModifier
{
public:
    SOFA_CLASS(QuadSetTopologyModifier,EdgeSetTopologyModifier);

    template <class DataTypes>
    friend class QuadSetGeometryAlgorithms;

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

    ~QuadSetTopologyModifier() override {}
public:
    void init() override;

    /** \brief add a set of quads
    @param quads an array of vertex indices describing the quads to be created
    */
    virtual void addQuads(const sofa::type::vector< Quad > &quads);

    /** \brief add a set of quads
    @param quads an array of vertex indices describing the quads to be created
    @param ancestors for each quad to be created provides an array of quad ancestors (optional)
    @param baryCoefs for each quad provides the barycentric coordinates (sum to 1) associated with each ancestor (optional)
    *
    */
    virtual void addQuads(const sofa::type::vector< Quad > &quads,
            const sofa::type::vector< sofa::type::vector< QuadID > > & ancestors,
            const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs) ;

    /** \brief Effectively Add a quad.
    */
    void addQuadProcess(Quad e);

     /** \brief Remove a set  of quads
    @param quads an array of quad indices to be removed (note that the array is not const since it needs to be sorted)
    *
    @param removeIsolatedEdges if true isolated edges are also removed
    @param removeIsolatedPoints if true isolated vertices are also removed
    *
    */
    virtual void removeQuads(const sofa::type::vector<QuadID> &quadIds,
            const bool removeIsolatedEdges,
            const bool removeIsolatedPoints);

    /** \brief Generic method to remove a list of items.
    */
    void removeItems(const sofa::type::vector< QuadID >& items) override;

protected:
    /** \brief Sends a message to warn that some quads were added in this topology.
    *
    * \sa addQuadsProcess
    */
    void addQuadsWarning(const size_t nQuads,
        const sofa::type::vector< Quad >& quadsList,
        const sofa::type::vector< QuadID >& quadsIndexList);

    /** \brief Sends a message to warn that some quads were added in this topology.
    *
    * \sa addQuadsProcess
    */
    void addQuadsWarning(const size_t nQuads,
        const sofa::type::vector< Quad >& quadsList,
        const sofa::type::vector< QuadID >& quadsIndexList,
        const sofa::type::vector< sofa::type::vector< QuadID > >& ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs);

    /** \brief Effectively Add some quads to this topology.
    *
     * \sa addQuadsWarning
     */
    virtual void addQuadsProcess(const sofa::type::vector< Quad >& quads);

    /** \brief Sends a message to warn that some quads are about to be deleted.
    *
    * \sa removeQuadsProcess
    *
    * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
    */
    virtual void removeQuadsWarning(sofa::type::vector<QuadID>& quads);

    /** \brief Remove a subset of  quads. Eventually remove isolated edges and vertices
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * \sa removeQuadsWarning
    *
    * @param removeIsolatedEdges if true isolated edges are also removed
    * @param removeIsolatedPoints if true isolated vertices are also removed
    */
    virtual void removeQuadsProcess(const sofa::type::vector<QuadID>& indices,
        const bool removeIsolatedEdges = false,
        const bool removeIsolatedPoints = false);


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
    * @param removeIsolatedItems if true isolated vertices are also removed
    * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
    */
    void removeEdgesProcess(const sofa::type::vector<QuadID>& indices,
        const bool removeIsolatedItems = false) override;


    /** \brief Add some points to this topology.
    *
    * \sa addPointsWarning
    */
    void addPointsProcess(const sofa::Size nPoints) override;

    /** \brief Remove a subset of points
    *
    * Elements corresponding to these points are removed from the mechanical object's state vectors.
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removePointsWarning before calling removePointsProcess.
    * \sa removePointsWarning
    * Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    */
    void removePointsProcess(const sofa::type::vector<PointID>& indices,
        const bool removeDOF = true) override;

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
    QuadSetTopologyContainer* 	m_container;
};

} //namespace sofa::component::topology::container::dynamic
