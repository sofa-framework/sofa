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
class TriangleSetTopologyContainer;

template <class DataTypes>
class TriangleSetGeometryAlgorithms;

/**
 * A class that modifies the topology by adding and removing triangles
 */
class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API TriangleSetTopologyModifier : public EdgeSetTopologyModifier
{
public:
    SOFA_CLASS(TriangleSetTopologyModifier,EdgeSetTopologyModifier);

    template <class DataTypes>
    friend class TriangleSetGeometryAlgorithms;

    typedef core::topology::BaseMeshTopology::TriangleID TriangleID;
    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef core::topology::BaseMeshTopology::TrianglesAroundVertex TrianglesAroundVertex;
    typedef core::topology::BaseMeshTopology::TrianglesAroundEdge TrianglesAroundEdge;
    typedef core::topology::BaseMeshTopology::EdgesInTriangle EdgesInTriangle;
protected:
    TriangleSetTopologyModifier()
        : list_Out(initData(&list_Out,"list_Out","triangles with at least one null values."))
    {}

    ~TriangleSetTopologyModifier() override {}
public:
    void init() override;

    void reinit() override;

    /** \brief add a set of triangles
    @param triangles an array of vertex indices describing the triangles to be created
     * Test precondition and apply:
     * TriangleSetTopologyModifier::addTrianglesProcess
     * TriangleSetTopologyModifier::addTrianglesPostProcessing
    */
    virtual void addTriangles(const sofa::type::vector< Triangle > &triangles);

    /** \brief add a set of triangles
    @param triangles an array of vertex indices describing the triangles to be created
    @param ancestors for each triangle to be created provides an array of triangle ancestors (optional)
    @param baryCoefs for each triangle provides the barycentric coordinates (sum to 1) associated with each ancestor (optional)
    *
    */
    virtual void addTriangles(const sofa::type::vector< Triangle > &triangles,
            const sofa::type::vector< sofa::type::vector< TriangleID > > & ancestors,
            const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs) ;

    /** \brief Effectively add a triangle to the topology.
    */
    void addTriangleProcess(Triangle t);

    /** \brief Generic method to remove a list of items.
     */
    void removeItems(const sofa::type::vector< TriangleID >& items) override;

    /** \brief Remove a set  of triangles
        @param triangles an array of triangle indices to be removed (note that the array is not const since it needs to be sorted)
        *
        @param removeIsolatedEdges if true isolated edges are also removed
        @param removeIsolatedPoints if true isolated vertices are also removed
        *
        */
    virtual void removeTriangles(const sofa::type::vector< TriangleID >& triangleIds,
            const bool removeIsolatedEdges,
            const bool removeIsolatedPoints);

    /** \brief Add and remove a subset of triangles. Eventually remove isolated edges and vertices
     *
     * This function is a complete workflow using differents methods of this class:
     * \sa removeTrianglesWarning
     * \sa removeTrianglesProcess
     * \sa addTrianglesProcess
     * \sa addTrianglesWarning
     *
     * @param nTri2Add - number of triangles to add.
     * @param triangles2Add - list of Triangle to add.
     * @param trianglesIndexList - List of their index.
     * @param ancestors - list of ancestors to these new triangles.
     * @param baryCoefs - their barycoefs related to these ancestors.
     * @param trianglesIndex2remove - List of triangle indices to remove.
     */
    virtual void addRemoveTriangles(const sofa::Size nTri2Add,
            const sofa::type::vector< Triangle >& triangles2Add,
            const sofa::type::vector< TriangleID >& trianglesIndex2Add,
            const sofa::type::vector< sofa::type::vector< TriangleID > > & ancestors,
            const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs,
            sofa::type::vector< TriangleID >& trianglesIndex2remove);


    /** \brief Duplicates the given edge. Only works if at least one of its points is adjacent to a border.
     * @returns the number of newly created points, or -1 if the incision failed.
     */
    virtual int InciseAlongEdge(EdgeID edge, int* createdPoints = nullptr);

protected:
    /** \brief Sends a message to warn that some triangles were added in this topology.
     *
     * \sa addTrianglesProcess
     */
    void addTrianglesWarning(const sofa::Size nTriangles,
        const sofa::type::vector< Triangle >& trianglesList,
        const sofa::type::vector< TriangleID >& trianglesIndexList);

    /** \brief Sends a message to warn that some triangles were added in this topology.
     *
     * \sa addTrianglesProcess
     */
    void addTrianglesWarning(const sofa::Size nTriangles,
        const sofa::type::vector< Triangle >& trianglesList,
        const sofa::type::vector< TriangleID >& trianglesIndexList,
        const sofa::type::vector< sofa::type::vector< TriangleID > >& ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs);

    /** \brief Effectively Add some triangles. Test precondition and apply:
    */
    virtual void addTrianglesProcess(const sofa::type::vector< Triangle >& triangles);

    /** \brief Sends a message to warn that some triangles are about to be deleted.
     *
     * \sa removeTrianglesProcess
     *
     * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
     */
    virtual void removeTrianglesWarning(sofa::type::vector<TriangleID>& triangles);

    /** \brief Remove a subset of  triangles. Eventually remove isolated edges and vertices
     *
     * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
     * \sa removeTrianglesWarning
     *
     * @param removeIsolatedEdges if true isolated edges are also removed
     * @param removeIsolatedPoints if true isolated vertices are also removed
     */
    virtual void removeTrianglesProcess(const sofa::type::vector<TriangleID>& indices,
        const bool removeIsolatedEdges = false,
        const bool removeIsolatedPoints = false);


    /** \brief Sends a message to warn that some edges were added in this topology.
     *
     * \sa addEdgesProcess
     */
    void addEdgesWarning(const sofa::Size nEdges,
        const sofa::type::vector< Edge >& edgesList,
        const sofa::type::vector< EdgeID >& edgesIndexList) override
    {
        EdgeSetTopologyModifier::addEdgesWarning(nEdges, edgesList, edgesIndexList);
    }

    /** \brief Sends a message to warn that some edges were added in this topology.
     *
     * \sa addEdgesProcess
     */
    void addEdgesWarning(const sofa::Size nEdges,
        const sofa::type::vector< Edge >& edgesList,
        const sofa::type::vector< EdgeID >& edgesIndexList,
        const sofa::type::vector< sofa::type::vector< EdgeID > >& ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs) override
    {
        EdgeSetTopologyModifier::addEdgesWarning(nEdges, edgesList, edgesIndexList, ancestors, baryCoefs);
    }

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
    void removeEdgesProcess(const sofa::type::vector<EdgeID>& indices,
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

    /** \brief Move input points indices to input new coords.
     * Also propagate event and update edgesAroundVertex and trianglesAroundVertex for data handling.
     *
     * @param id : list of indices to move
     * @param : ancestors list of ancestors to define relative new position
     * @param coefs : barycoef to locate new coord relatively to ancestors.
     * @moveDOF bool allowing the move (default true)
     */
    void movePointsProcess(const sofa::type::vector< PointID >& id,
        const sofa::type::vector< sofa::type::vector< PointID > >& ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& coefs,
        const bool moveDOF = true) override;

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



    /** \brief Precondition to fulfill before removing triangles. No preconditions are needed in this class. This function should be inplemented in children classes.
     *
     */
    virtual bool removeTrianglesPreconditions(const sofa::type::vector< TriangleID >& items);


    /**\brief: Postprocessing to apply to topology triangles. Nothing to do in this class. This function should be inplemented in children classes.
     *
     */
    virtual void removeTrianglesPostProcessing(const sofa::type::vector< TriangleID >& edgeToBeRemoved, const sofa::type::vector< TriangleID >& vertexToBeRemoved );


    /** \brief Precondition to fulfill before adding triangles. No preconditions are needed in this class. This function should be inplemented in children classes.
     *
     */
    virtual bool addTrianglesPreconditions(const sofa::type::vector<Triangle>& triangles);


    /**\brief: Postprocessing to apply to topology triangles. Nothing to do in this class. This function should be inplemented in children classes.
     *
     */
    virtual void addTrianglesPostProcessing(const sofa::type::vector<Triangle>& triangles);

    Data<sofa::type::vector<TriangleID> > list_Out; ///< triangles with at least one null values.
private:
    TriangleSetTopologyContainer*	m_container;
};

} //namespace sofa::component::topology::container::dynamic
