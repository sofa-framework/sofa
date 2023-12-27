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

#include <sofa/component/topology/mapping/config.h>
#include <sofa/core/topology/TopologicalMapping.h>

#include <sofa/type/Vec.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <map>

#include <sofa/core/BaseMapping.h>
#include <sofa/core/behavior/MechanicalState.h>


namespace sofa::component::topology::container::dynamic
{
    class QuadSetTopologyContainer;
    class QuadSetTopologyModifier;
}

namespace sofa::component::topology::mapping
{

/**
* This class, called Edge2QuadTopologicalMapping, is a specific implementation of the interface TopologicalMapping where :
*
* INPUT TOPOLOGY = EdgeSetTopology
* OUTPUT TOPOLOGY = QuadSetTopology based on new DOFs, as the tubular skinning of INPUT TOPOLOGY.
*
* Edge2QuadTopologicalMapping class is templated by the pair (INPUT TOPOLOGY, OUTPUT TOPOLOGY)
*
*/
class SOFA_COMPONENT_TOPOLOGY_MAPPING_API Edge2QuadTopologicalMapping : public sofa::core::topology::TopologicalMapping
{
public:
    SOFA_CLASS(Edge2QuadTopologicalMapping,sofa::core::topology::TopologicalMapping);

    using RigidCoord = sofa::core::State<defaulttype::Rigid3Types>::Coord;
    using Vec3Coord = sofa::core::State<defaulttype::Vec3Types>::Coord;
    
    using Index = sofa::core::topology::BaseMeshTopology::Index;
    using Edge = sofa::core::topology::BaseMeshTopology::Edge;
    using Quad = sofa::core::topology::BaseMeshTopology::Quad;
    using VecIndex = type::vector<Index>;

protected:
    /** \brief Constructor.
    *
    * @param from the topology issuing TopologyChange objects (the "source").
    * @param to   the topology for which the TopologyChange objects must be translated (the "target").
    */
    Edge2QuadTopologicalMapping();

    /** \brief Destructor.
    *
    * Does nothing.
    */
    ~Edge2QuadTopologicalMapping() override
    {}

public:
    /** \brief Initializes the target BaseTopology from the source BaseTopology.
    */
    void init() override;

    /** \brief Translates the TopologyChange objects from the source to the target.
    *
    * Translates each of the TopologyChange objects waiting in the source list so that they have a meaning and
    * reflect the effects of the first topology changes on the second topology.
    *
    */
    void updateTopologicalMappingTopDown() override;

    Index getFromIndex(Index ind) override;


    Data<unsigned int> d_nbPointsOnEachCircle; ///< number of points to create along the circles around each point of the input topology (10 by default)
    Data<SReal> d_radius; ///< radius of the circles around each point of the input topology (1 by default)
    Data<SReal> d_radiusFocal; ///< in case of ellipse, (extra) radius on the focal axis (0 by default)
    Data<type::Vec3> d_focalAxis; ///< in case of ellipse, focal axis (default [0,0,1])

    Data<VecIndex> d_edgeList; ///< list of input edges for the topological mapping: by default, all considered
    Data<bool> d_flipNormals; ///< Flip Normal ? (Inverse point order when creating quad)

    SingleLink<Edge2QuadTopologicalMapping, topology::container::dynamic::QuadSetTopologyContainer, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_toQuadContainer; ///< Output container storing Quads
    SingleLink<Edge2QuadTopologicalMapping, topology::container::dynamic::QuadSetTopologyModifier, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_toQuadModifier; ///< Output modifier handling Quads
};

} //namespace sofa::component::topology::mapping
