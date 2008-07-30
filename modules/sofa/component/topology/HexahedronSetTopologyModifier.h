/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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

#include <sofa/component/topology/QuadSetTopologyModifier.h>

namespace sofa
{
namespace component
{
namespace topology
{
class HexahedronSetTopologyContainer;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::HexaID HexaID;
typedef BaseMeshTopology::Hexa Hexa;
typedef BaseMeshTopology::SeqHexas SeqHexas;
typedef BaseMeshTopology::VertexHexas VertexHexas;
typedef BaseMeshTopology::EdgeHexas EdgeHexas;
typedef BaseMeshTopology::QuadHexas QuadHexas;
typedef BaseMeshTopology::HexaEdges HexaEdges;
typedef BaseMeshTopology::HexaQuads HexaQuads;

typedef Hexa Hexahedron;
typedef HexaEdges HexahedronEdges;
typedef HexaQuads HexahedronQuads;

/**
* A class that modifies the topology by adding and removing hexahedra
*/
class HexahedronSetTopologyModifier : public QuadSetTopologyModifier
{
public:
    HexahedronSetTopologyModifier()
        : QuadSetTopologyModifier()
    { }

    HexahedronSetTopologyModifier(core::componentmodel::topology::TopologyContainer *container)
        : QuadSetTopologyModifier(container)
    { }

    virtual ~HexahedronSetTopologyModifier() {}

    HexahedronSetTopologyContainer* getHexahedronSetTopologyContainer() const;

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
    void removeHexahedraWarning( sofa::helper::vector<unsigned int> &hexahedra);

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
    * Use a list of ancestors to create the new points.
    * Last parameter baryCoefs defines the coefficient used for the creation of the new points.
    * Default value for these coefficient (when none is defined) is 1/n with n being the number of ancestors
    * for the point being created.
    * Important : the points are actually added to the mechanical object's state vectors iff (addDOF == true)
    *
    * \sa addPointsWarning
    */
    virtual void addPointsProcess(const unsigned int nPoints, const bool addDOF = true);

    /** \brief Add some points to this topology.
    *
    * Use a list of ancestors to create the new points.
    * Last parameter baryCoefs defines the coefficient used for the creation of the new points.
    * Default value for these coefficient (when none is defined) is 1/n with n being the number of ancestors
    * for the point being created.
    * Important : the points are actually added to the mechanical object's state vectors iff (addDOF == true)
    *
    * \sa addPointsWarning
    */
    virtual void addPointsProcess(const unsigned int nPoints,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
            const bool addDOF = true);

    /** \brief Remove a subset of points
    *
    * Elements corresponding to these points are removed form the mechanical object's state vectors.
    *
    * Important : some structures might need to be warned BEFORE the points are actually deleted, so always use method removePointsWarning before calling removePointsProcess.
    * \sa removePointsWarning
    * Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    */
    virtual void removePointsProcess( sofa::helper::vector<unsigned int> &indices, const bool removeDOF = true);

    /** \brief Reorder this topology.
    *
    * Important : the points are actually renumbered in the mechanical object's state vectors iff (renumberDOF == true)
    * \see MechanicalObject::renumberValues
    */
    virtual void renumberPointsProcess( const sofa::helper::vector<unsigned int> &index,
            const sofa::helper::vector<unsigned int>& inv_index,
            const bool renumberDOF = true);

protected:
    /** \brief Add a hexahedron.
    */
    void addHexahedron(Hexahedron e);
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
