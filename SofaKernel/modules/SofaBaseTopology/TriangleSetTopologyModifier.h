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
#ifndef SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYMODIFIER_H
#define SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYMODIFIER_H
#include "config.h"

#include <SofaBaseTopology/EdgeSetTopologyModifier.h>

namespace sofa
{
namespace component
{
namespace topology
{
class TriangleSetTopologyContainer;


/**
 * A class that modifies the topology by adding and removing triangles
 */
class SOFA_BASE_TOPOLOGY_API TriangleSetTopologyModifier : public EdgeSetTopologyModifier
{
public:
    SOFA_CLASS(TriangleSetTopologyModifier,EdgeSetTopologyModifier);

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

    virtual ~TriangleSetTopologyModifier() override {}
public:
    virtual void init() override;

    virtual void reinit() override;

    /// \brief function to propagate topological change events by parsing the list of topologyEngines linked to this topology.
    virtual void propagateTopologicalEngineChanges() override;

    /** \brief add a set of triangles
    @param triangles an array of vertex indices describing the triangles to be created
     * Test precondition and apply:
     * TriangleSetTopologyModifier::addTrianglesProcess
     * TriangleSetTopologyModifier::addTrianglesPostProcessing
    */
    virtual void addTriangles(const sofa::helper::vector< Triangle > &triangles);

    /** \brief add a set of triangles
    @param triangles an array of vertex indices describing the triangles to be created
    @param ancestors for each triangle to be created provides an array of triangle ancestors (optional)
    @param baryCoefs for each triangle provides the barycentric coordinates (sum to 1) associated with each ancestor (optional)
    *
    */
    virtual void addTriangles(const sofa::helper::vector< Triangle > &triangles,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs) ;


    /** \brief Sends a message to warn that some triangles were added in this topology.
     *
     * \sa addTrianglesProcess
     */
    void addTrianglesWarning(const unsigned int nTriangles,
            const sofa::helper::vector< Triangle >& trianglesList,
            const sofa::helper::vector< unsigned int >& trianglesIndexList) ;

    /** \brief Sends a message to warn that some triangles were added in this topology.
     *
     * \sa addTrianglesProcess
     */
    void addTrianglesWarning(const unsigned int nTriangles,
            const sofa::helper::vector< Triangle >& trianglesList,
            const sofa::helper::vector< unsigned int >& trianglesIndexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs) ;

    /** \brief Effectively add a triangle to the topology.
     */
    void addTriangleProcess (Triangle t);

    /** \brief Effectively Add some triangles. Test precondition and apply:
    */
    virtual void addTrianglesProcess(const sofa::helper::vector< Triangle > &triangles);

    /** \brief Add some points to this topology.
     *
     * \sa addPointsWarning
     */
    virtual void addPointsProcess(const unsigned int nPoints) override;

    /** \brief Sends a message to warn that some edges were added in this topology.
     *
     * \sa addEdgesProcess
     */
    void addEdgesWarning(const unsigned int nEdges,
            const sofa::helper::vector< Edge >& edgesList,
            const sofa::helper::vector< unsigned int >& edgesIndexList) override
    {
        EdgeSetTopologyModifier::addEdgesWarning( nEdges, edgesList, edgesIndexList);
    }

    /** \brief Sends a message to warn that some edges were added in this topology.
     *
     * \sa addEdgesProcess
     */
    void addEdgesWarning(const unsigned int nEdges,
            const sofa::helper::vector< Edge >& edgesList,
            const sofa::helper::vector< unsigned int >& edgesIndexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs) override
    {
        EdgeSetTopologyModifier::addEdgesWarning( nEdges, edgesList, edgesIndexList, ancestors, baryCoefs);
    }

    /** \brief Add some edges to this topology.
     *
     * \sa addEdgesWarning
     */
    void addEdgesProcess(const sofa::helper::vector< Edge > &edges) override;


    /** \brief Generic method to remove a list of items.
     */
    virtual void removeItems(const sofa::helper::vector< unsigned int >& items) override;

    /** \brief Remove a set  of triangles
        @param triangles an array of triangle indices to be removed (note that the array is not const since it needs to be sorted)
        *
        @param removeIsolatedEdges if true isolated edges are also removed
        @param removeIsolatedPoints if true isolated vertices are also removed
        *
        */
    virtual void removeTriangles(const sofa::helper::vector< unsigned int >& triangleIds,
            const bool removeIsolatedEdges,
            const bool removeIsolatedPoints);


    /** \brief Sends a message to warn that some triangles are about to be deleted.
     *
     * \sa removeTrianglesProcess
     *
     * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
     */
    virtual void removeTrianglesWarning(sofa::helper::vector<unsigned int> &triangles);


    /** \brief Remove a subset of  triangles. Eventually remove isolated edges and vertices
     *
     * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
     * \sa removeTrianglesWarning
     *
     * @param removeIsolatedEdges if true isolated edges are also removed
     * @param removeIsolatedPoints if true isolated vertices are also removed
     */
    virtual void removeTrianglesProcess( const sofa::helper::vector<unsigned int> &indices,
            const bool removeIsolatedEdges=false,
            const bool removeIsolatedPoints=false);


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
    virtual void addRemoveTriangles(const unsigned int nTri2Add,
            const sofa::helper::vector< Triangle >& triangles2Add,
            const sofa::helper::vector< unsigned int >& trianglesIndex2Add,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
            sofa::helper::vector< unsigned int >& trianglesIndex2remove);



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
    virtual void renumberPointsProcess( const sofa::helper::vector<unsigned int> &index,
            const sofa::helper::vector<unsigned int> &inv_index,
            const bool renumberDOF = true) override;


    /** \brief Generic method for points renumbering
     */
    virtual void renumberPoints( const sofa::helper::vector<unsigned int> &index,
            const sofa::helper::vector<unsigned int> &inv_index) override;


    /** \brief Move input points indices to input new coords.
     * Also propagate event and update edgesAroundVertex and trianglesAroundVertex for data handling.
     *
     * @param id : list of indices to move
     * @param : ancestors list of ancestors to define relative new position
     * @param coefs : barycoef to locate new coord relatively to ancestors.
     * @moveDOF bool allowing the move (default true)
     */
    virtual void movePointsProcess (const sofa::helper::vector <unsigned int>& id,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs,
            const bool moveDOF = true) override;


protected:

    /** \brief Precondition to fulfill before removing triangles. No preconditions are needed in this class. This function should be inplemented in children classes.
     *
     */
    virtual bool removeTrianglesPreconditions(const sofa::helper::vector< unsigned int >& items);


    /**\brief: Postprocessing to apply to topology triangles. Nothing to do in this class. This function should be inplemented in children classes.
     *
     */
    virtual void removeTrianglesPostProcessing(const sofa::helper::vector< unsigned int >& edgeToBeRemoved, const sofa::helper::vector< unsigned int >& vertexToBeRemoved );


    /** \brief Precondition to fulfill before adding triangles. No preconditions are needed in this class. This function should be inplemented in children classes.
     *
     */
    virtual bool addTrianglesPreconditions(const sofa::helper::vector <Triangle>& triangles);


    /**\brief: Postprocessing to apply to topology triangles. Nothing to do in this class. This function should be inplemented in children classes.
     *
     */
    virtual void addTrianglesPostProcessing(const sofa::helper::vector <Triangle>& triangles);

    Data<sofa::helper::vector <unsigned int> > list_Out;
private:
    TriangleSetTopologyContainer*	m_container;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
