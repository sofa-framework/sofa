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
#ifndef SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGYMODIFIER_H
#define SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGYMODIFIER_H

#include <SofaBaseTopology/QuadSetTopologyModifier.h>

namespace sofa
{
namespace component
{
namespace topology
{
class HexahedronSetTopologyContainer;

using core::topology::BaseMeshTopology;
typedef BaseMeshTopology::HexaID HexaID;
typedef BaseMeshTopology::Hexa Hexa;
typedef BaseMeshTopology::SeqHexahedra SeqHexahedra;
typedef BaseMeshTopology::HexahedraAroundVertex HexahedraAroundVertex;
typedef BaseMeshTopology::HexahedraAroundEdge HexahedraAroundEdge;
typedef BaseMeshTopology::HexahedraAroundQuad HexahedraAroundQuad;
typedef BaseMeshTopology::EdgesInHexahedron EdgesInHexahedron;
typedef BaseMeshTopology::QuadsInHexahedron QuadsInHexahedron;

typedef Hexa Hexahedron;
typedef EdgesInHexahedron EdgesInHexahedron;
typedef QuadsInHexahedron QuadsInHexahedron;

/**
* A class that modifies the topology by adding and removing hexahedra
*/
class SOFA_BASE_TOPOLOGY_API HexahedronSetTopologyModifier : public QuadSetTopologyModifier
{
public:
    SOFA_CLASS(HexahedronSetTopologyModifier,QuadSetTopologyModifier);
protected:
    HexahedronSetTopologyModifier()
        : QuadSetTopologyModifier()
    { }

    virtual ~HexahedronSetTopologyModifier() {}
public:
    virtual void init();

    /// \brief function to propagate topological change events by parsing the list of topologyEngines linked to this topology.
    virtual void propagateTopologicalEngineChanges();


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
    virtual void addQuadsProcess(const sofa::helper::vector< Quad > &quads);

    /** \brief Remove a subset of quads
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removeEdgesWarning before calling removeEdgesProcess.
    * @param removeIsolatedEdges if true isolated edges are also removed
    * @param removeIsolatedPoints if true isolated vertices are also removed
    */
    virtual void removeQuadsProcess(const sofa::helper::vector<unsigned int> &indices,
            const bool removeIsolatedEdges = false,
            const bool removeIsolatedPoints = false);

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
    virtual void removeEdgesProcess(const sofa::helper::vector<unsigned int> &indices,
            const bool removeIsolatedItems = false);

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
            const sofa::helper::vector<unsigned int>& inv_index,
            const bool renumberDOF = true);

    /** \brief Remove a set  of hexahedra
    @param hexahedra an array of hexahedron indices to be removed (note that the array is not const since it needs to be sorted)
    *
    */
    virtual void removeHexahedra(sofa::helper::vector< unsigned int >& hexahedra);

    /** \brief Generic method to remove a list of items.
    */
    virtual void removeItems(sofa::helper::vector< unsigned int >& items);

    /** \brief Generic method for points renumbering
    */
    virtual void renumberPoints( const sofa::helper::vector<unsigned int>& index,
            const sofa::helper::vector<unsigned int>& inv_index);


private:
    HexahedronSetTopologyContainer* 	m_container;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
