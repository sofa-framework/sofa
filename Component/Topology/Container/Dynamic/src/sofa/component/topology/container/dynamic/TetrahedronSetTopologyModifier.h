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

#include <sofa/component/topology/container/dynamic/TriangleSetTopologyModifier.h>

namespace sofa::component::topology::container::dynamic
{
class TetrahedronSetTopologyContainer;

template <class DataTypes>
class TetrahedronSetGeometryAlgorithms;

/**
* A class that modifies the topology by adding and removing tetrahedra
*/
class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API TetrahedronSetTopologyModifier : public TriangleSetTopologyModifier
{
public:
    SOFA_CLASS(TetrahedronSetTopologyModifier,TriangleSetTopologyModifier);

    template <class DataTypes>
    friend class TetrahedronSetGeometryAlgorithms;

    typedef core::topology::BaseMeshTopology::TetraID TetraID;
    typedef core::topology::BaseMeshTopology::TetrahedronID TetrahedronID;
    typedef core::topology::BaseMeshTopology::Tetra Tetra;
    typedef core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;
    typedef core::topology::BaseMeshTopology::TetrahedraAroundVertex TetrahedraAroundVertex;
    typedef core::topology::BaseMeshTopology::TetrahedraAroundEdge TetrahedraAroundEdge;
    typedef core::topology::BaseMeshTopology::TetrahedraAroundTriangle TetrahedraAroundTriangle;
    typedef core::topology::BaseMeshTopology::EdgesInTetrahedron EdgesInTetrahedron;
    typedef core::topology::BaseMeshTopology::TrianglesInTetrahedron TrianglesInTetrahedron;
    typedef Tetra Tetrahedron;


    Data< bool > removeIsolated; ///< Controlled DOF index.
protected:
    TetrahedronSetTopologyModifier()
        : TriangleSetTopologyModifier()
        , removeIsolated( initData(&removeIsolated,true, "removeIsolated", "remove Isolated dof") )
    {}

    ~TetrahedronSetTopologyModifier() override {}
public:
    void init() override;

    void reinit() override;

    /** \brief add a set of tetrahedra
    @param tetrahedra an array of vertex indices describing the tetrahedra to be created
    */
    virtual void addTetrahedra(const sofa::type::vector< Tetrahedron > &tetrahedra);

    /** \brief add a set of tetrahedra
    @param quads an array of vertex indices describing the tetrahedra to be created
    @param ancestors for each tetrahedron to be created provides an array of tetrahedron ancestors (optional)
    @param baryCoefs for each tetrahedron provides the barycentric coordinates (sum to 1) associated with each ancestor (optional)
    *
    */
    virtual void addTetrahedra(const sofa::type::vector< Tetrahedron > &tetrahedra,
            const sofa::type::vector< sofa::type::vector< TetrahedronID > > & ancestors,
            const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs) ;

    /** \brief Add a tetrahedron.
    *
    */
    void addTetrahedronProcess(Tetrahedron e);
    
    /** \brief Remove a set  of tetrahedra
    @param tetrahedra an array of tetrahedron indices to be removed (note that the array is not const since it needs to be sorted)
    *
    */
    virtual void removeTetrahedra(const sofa::type::vector< TetrahedronID >& tetrahedraIds, const bool removeIsolatedItems = true);

    /** \brief Generic method to remove a list of items.
    */
    void removeItems(const sofa::type::vector<TetrahedronID> &items) override;

    /** \brief  Removes all tetrahedra in the ball of center "ind_ta" and of radius dist(ind_ta, ind_tb)
    */
    void RemoveTetraBall(TetrahedronID ind_ta, TetrahedronID ind_tb);

protected:
    /** \brief Sends a message to warn that some tetrahedra were added in this topology.
    *
    * \sa addTetrahedraProcess
    */
    void addTetrahedraWarning(const size_t nTetrahedra,
        const sofa::type::vector< Tetrahedron >& tetrahedraList,
        const sofa::type::vector< TetrahedronID >& tetrahedraIndexList);

    /** \brief Sends a message to warn that some tetrahedra were added in this topology.
    *
    * \sa addTetrahedraProcess
    */
    void addTetrahedraWarning(const size_t nTetrahedra,
        const sofa::type::vector< Tetrahedron >& tetrahedraList,
        const sofa::type::vector< TetrahedronID >& tetrahedraIndexList,
        const sofa::type::vector< sofa::type::vector< TetrahedronID > >& ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs);

    /** \brief Actually Add some tetrahedra to this topology.
    *
    * \sa addTetrahedraWarning
    */
    virtual void addTetrahedraProcess(const sofa::type::vector< Tetrahedron >& tetrahedra);

    /** \brief Sends a message to warn that some tetrahedra are about to be deleted.
    *
    * \sa removeTetrahedraProcess
    *
    * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
    */
    void removeTetrahedraWarning(sofa::type::vector<TetrahedronID>& tetrahedra);

    /** \brief Remove a subset of tetrahedra
    *
    * Elements corresponding to these points are removed form the mechanical object's state vectors.
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * \sa removeTetrahedraWarning
    * @param removeIsolatedItems if true remove isolated triangles, edges and vertices
    */
    virtual void removeTetrahedraProcess(const sofa::type::vector<TetrahedronID>& indices,
        const bool removeIsolatedItems = false);



    /** \brief Actually Add some triangles to this topology.
    *
    * \sa addTrianglesWarning
    */
    void addTrianglesProcess(const sofa::type::vector< Triangle >& triangles) override;

    /** \brief Remove a subset of triangles
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * @param removeIsolatedEdges if true isolated edges are also removed
    * @param removeIsolatedPoints if true isolated vertices are also removed
    */
    void removeTrianglesProcess(const sofa::type::vector<TriangleID>& indices,
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
        const sofa::type::vector<PointID>&/*inv_index*/,
        const bool renumberDOF = true) override;


    /// \brief function to propagate topological change events by parsing the list of TopologyHandlers linked to this topology.
    void propagateTopologicalEngineChanges() override;

private:
    TetrahedronSetTopologyContainer* 	m_container;
};

} //namespace sofa::component::topology::container::dynamic
