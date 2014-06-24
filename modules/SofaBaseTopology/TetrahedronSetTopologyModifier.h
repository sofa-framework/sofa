/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGYMODIFIER_H
#define SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGYMODIFIER_H

#include <SofaBaseTopology/TriangleSetTopologyModifier.h>

namespace sofa
{

namespace component
{

namespace topology
{
class TetrahedronSetTopologyContainer;

using core::topology::BaseMeshTopology;
typedef BaseMeshTopology::TetraID TetraID;
typedef BaseMeshTopology::Tetra Tetra;
typedef BaseMeshTopology::SeqTetrahedra SeqTetrahedra;
typedef BaseMeshTopology::TetrahedraAroundVertex TetrahedraAroundVertex;
typedef BaseMeshTopology::TetrahedraAroundEdge TetrahedraAroundEdge;
typedef BaseMeshTopology::TetrahedraAroundTriangle TetrahedraAroundTriangle;
typedef BaseMeshTopology::EdgesInTetrahedron EdgesInTetrahedron;
typedef BaseMeshTopology::TrianglesInTetrahedron TrianglesInTetrahedron;

typedef Tetra Tetrahedron;
typedef EdgesInTetrahedron EdgesInTetrahedron;
typedef TrianglesInTetrahedron TrianglesInTetrahedron;

/**
* A class that modifies the topology by adding and removing tetrahedra
*/
class SOFA_BASE_TOPOLOGY_API TetrahedronSetTopologyModifier : public TriangleSetTopologyModifier
{
public:
    SOFA_CLASS(TetrahedronSetTopologyModifier,TriangleSetTopologyModifier);

    Data< bool > removeIsolated; ///< Controlled DOF index.
protected:
    TetrahedronSetTopologyModifier()
        : TriangleSetTopologyModifier()
        , removeIsolated( initData(&removeIsolated,true, "removeIsolated", "remove Isolated dof") )
    {}

    virtual ~TetrahedronSetTopologyModifier() {}
public:
    virtual void init();

    virtual void reinit();

    /// \brief function to propagate topological change events by parsing the list of topologyEngines linked to this topology.
    virtual void propagateTopologicalEngineChanges();

    /** \brief add a set of tetrahedra
    @param tetrahedra an array of vertex indices describing the tetrahedra to be created
    */
    virtual void addTetrahedra(const sofa::helper::vector< Tetrahedron > &tetrahedra);

    /** \brief add a set of tetrahedra
    @param quads an array of vertex indices describing the tetrahedra to be created
    @param ancestors for each tetrahedron to be created provides an array of tetrahedron ancestors (optional)
    @param baryCoefs for each tetrahedron provides the barycentric coordinates (sum to 1) associated with each ancestor (optional)
    *
    */
    virtual void addTetrahedra(const sofa::helper::vector< Tetrahedron > &tetrahedra,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs) ;


    /** \brief Sends a message to warn that some tetrahedra were added in this topology.
    *
    * \sa addTetrahedraProcess
    */
    void addTetrahedraWarning(const unsigned int nTetrahedra,
            const sofa::helper::vector< Tetrahedron >& tetrahedraList,
            const sofa::helper::vector< unsigned int >& tetrahedraIndexList);

    /** \brief Sends a message to warn that some tetrahedra were added in this topology.
    *
    * \sa addTetrahedraProcess
    */
    void addTetrahedraWarning(const unsigned int nTetrahedra,
            const sofa::helper::vector< Tetrahedron >& tetrahedraList,
            const sofa::helper::vector< unsigned int >& tetrahedraIndexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs);

    /** \brief Add a tetrahedron.
    *
    */
    void addTetrahedronProcess(Tetrahedron e);

    /** \brief Actually Add some tetrahedra to this topology.
    *
    * \sa addTetrahedraWarning
    */
    virtual void addTetrahedraProcess(const sofa::helper::vector< Tetrahedron > &tetrahedra);

    /** \brief Sends a message to warn that some tetrahedra are about to be deleted.
    *
    * \sa removeTetrahedraProcess
    *
    * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
    */
    void removeTetrahedraWarning( sofa::helper::vector<unsigned int> &tetrahedra);

    /** \brief Remove a subset of tetrahedra
    *
    * Elements corresponding to these points are removed form the mechanical object's state vectors.
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * \sa removeTetrahedraWarning
    * @param removeIsolatedItems if true remove isolated triangles, edges and vertices
    */
    virtual void removeTetrahedraProcess( const sofa::helper::vector<unsigned int> &indices,
            const bool removeIsolatedItems=false);

    /** \brief Actually Add some triangles to this topology.
    *
    * \sa addTrianglesWarning
    */
    virtual void addTrianglesProcess(const sofa::helper::vector< Triangle > &triangles);

    /** \brief Remove a subset of triangles
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * @param removeIsolatedEdges if true isolated edges are also removed
    * @param removeIsolatedPoints if true isolated vertices are also removed
    */
    virtual void removeTrianglesProcess(const sofa::helper::vector<unsigned int> &indices,
            const bool removeIsolatedEdges=false,
            const bool removeIsolatedPoints=false);

    /** \brief Add some edges to this topology.
    *
    * \sa addEdgesWarning
    */
    virtual void addEdgesProcess(const sofa::helper::vector< Edge > &edges);

    /** \brief Remove a subset of edges
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * \sa removeEdgesWarning
    *
    * Important : parameter indices is not const because it is actually sorted from the highest index to the lowest one.
    * @param removeIsolatedItems if true remove isolated vertices
    */
    virtual void removeEdgesProcess( const sofa::helper::vector<unsigned int> &indices,
            const bool removeIsolatedItems=false);

    /** \brief Add some points to this topology.
    *
    * \sa addPointsWarning
    */
    virtual void addPointsProcess(const unsigned int nPoints);

    /** \brief Remove a subset of points
    *
    * Elements corresponding to these points are removed form the mechanical object's state vectors.
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removePointsWarning before calling removePointsProcess.
    * \sa removePointsWarning
    * Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    */
    virtual void removePointsProcess(const sofa::helper::vector<unsigned int> &indices, const bool removeDOF = true);

    /** \brief Reorder this topology.
    *
    * Important : the points are actually renumbered in the mechanical object's state vectors iff (renumberDOF == true)
    * \see MechanicalObject::renumberValues
    */
    virtual void renumberPointsProcess( const sofa::helper::vector<unsigned int> &index,
            const sofa::helper::vector<unsigned int> &/*inv_index*/,
            const bool renumberDOF = true);

    /** \brief Remove a set  of tetrahedra
    @param tetrahedra an array of tetrahedron indices to be removed (note that the array is not const since it needs to be sorted)
    *
    */
    virtual void removeTetrahedra(sofa::helper::vector< unsigned int >& tetrahedra);

    /** \brief Generic method to remove a list of items.
    */
    virtual void removeItems(sofa::helper::vector< unsigned int >& items);

    /** \brief  Removes all tetrahedra in the ball of center "ind_ta" and of radius dist(ind_ta, ind_tb)
    */
    void RemoveTetraBall(unsigned int ind_ta, unsigned int ind_tb);

    /** \brief Generic method for points renumbering
    */
    virtual void renumberPoints( const sofa::helper::vector<unsigned int> &/*index*/,
            const sofa::helper::vector<unsigned int> &/*inv_index*/);


private:
    TetrahedronSetTopologyContainer* 	m_container;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
