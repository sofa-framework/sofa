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
#include <map>

#include <sofa/core/BaseMapping.h>

namespace sofa::component::topology::mapping
{



/**
* This class, called Quad2TriangleTopologicalMapping, is a specific implementation of the interface TopologicalMapping where :
*
* INPUT TOPOLOGY = QuadSetTopology
* OUTPUT TOPOLOGY = TriangleSetTopology, as the constitutive elements of the INPUT TOPOLOGY
*
* Quad2TriangleTopologicalMapping class is templated by the pair (INPUT TOPOLOGY, OUTPUT TOPOLOGY)
*
*/

class SOFA_COMPONENT_TOPOLOGY_MAPPING_API Quad2TriangleTopologicalMapping : public sofa::core::topology::TopologicalMapping
{

public:
    SOFA_CLASS(Quad2TriangleTopologicalMapping,sofa::core::topology::TopologicalMapping);
protected:
    /** \brief Constructor.
    *
    */
    Quad2TriangleTopologicalMapping();

    /** \brief Destructor.
    *
    * Does nothing.
    */
    ~Quad2TriangleTopologicalMapping() override;
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

};

} //namespace sofa::component::topology::mapping
