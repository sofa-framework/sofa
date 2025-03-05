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
#include <sofa/component/mapping/linear/config.h>

#include <sofa/core/topology/TopologicalMapping.h>

#include <sofa/type/Vec.h>
#include <map>
#include <set>

#include <sofa/core/BaseMapping.h>
#include <sofa/core/topology/TopologyData.h>

namespace sofa::component::mapping::linear
{

/**
 * This class, called SimpleTesselatedTetraTopologicalMapping, is a specific implementation of the interface TopologicalMapping where :
 *
 * INPUT TOPOLOGY = HexahedronSetTopology
 * OUTPUT TOPOLOGY = Set of HexahedronSetTopologies, as the boundary of the INPUT TOPOLOGY
 *
 * Each hexahedron in the input Topology will be divided in eight hexahedrom in the output topology
 *
 * SimpleTesselatedHexaTopologicalMapping class is templated by the pair (INPUT TOPOLOGY, OUTPUT TOPOLOGY)
 *
*/

class SOFA_COMPONENT_MAPPING_LINEAR_API SimpleTesselatedHexaTopologicalMapping : public sofa::core::topology::TopologicalMapping
{
public:
    SOFA_CLASS(SimpleTesselatedHexaTopologicalMapping,sofa::core::topology::TopologicalMapping);
protected:
    /** \brief Constructor.
     *
     */
    SimpleTesselatedHexaTopologicalMapping();

    /** \brief Destructor.
     *
         * Does nothing.
         */
    ~SimpleTesselatedHexaTopologicalMapping() override {}
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
    void updateTopologicalMappingTopDown() override {};

    /** \brief Translates the TopologyChange objects from the source to the target.
     *
     * Translates each of the TopologyChange objects waiting in the source list so that they have a meaning and
     * reflect the effects of the first topology changes on the second topology.
     *
     */
protected:
    type::vector<sofa::Index> pointMappedFromPoint;
    std::map<type::fixed_array<sofa::Index,2>, sofa::Index> pointMappedFromEdge;
    std::map<type::fixed_array<sofa::Index,4>, sofa::Index> pointMappedFromFacet;
    type::vector<sofa::Index> pointMappedFromHexa;
};

} //namespace sofa::component::mapping::linear
