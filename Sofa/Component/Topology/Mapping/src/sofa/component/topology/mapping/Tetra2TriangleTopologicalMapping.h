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

namespace sofa::component::topology::container::dynamic
{
    class TriangleSetTopologyModifier;
} // namespace sofa::component::topology::container::dynamic

namespace sofa::component::topology::mapping
{

/**
 * This class, called Tetra2TriangleTopologicalMapping, is a specific implementation of the interface TopologicalMapping where :
 *
 * INPUT TOPOLOGY = TetrahedronSetTopology
 * OUTPUT TOPOLOGY = TriangleSetTopology, as the boundary of the INPUT TOPOLOGY
 *
 * Tetra2TriangleTopologicalMapping class is templated by the pair (INPUT TOPOLOGY, OUTPUT TOPOLOGY)
 *
*/
class SOFA_COMPONENT_TOPOLOGY_MAPPING_API Tetra2TriangleTopologicalMapping : public sofa::core::topology::TopologicalMapping
{
public:
    SOFA_CLASS(Tetra2TriangleTopologicalMapping,sofa::core::topology::TopologicalMapping);
protected:
    /** \brief Constructor.
     *
     */
    Tetra2TriangleTopologicalMapping();

    /** \brief Destructor.
     *
     * Does nothing.
     */
    ~Tetra2TriangleTopologicalMapping() override = default;
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

    /// Method to check the topology mapping maps regarding the upper topology
    bool checkTopologies() override;

protected:
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA()
    Data<bool> flipNormals;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA()
    Data<bool> noNewTriangles;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA()
    Data<bool> noInitialTriangles;


    Data<bool> d_flipNormals; ///< Flip Normal ? (Inverse point order when creating triangle)
    Data<bool> d_noNewTriangles; ///< If true no new triangles are being created
    Data<bool> d_noInitialTriangles; ///< If true the list of initial triangles is initially empty. Only additional triangles will be added in the list
    sofa::type::vector<Index> addedTriangleIndex;
    container::dynamic::TriangleSetTopologyModifier* m_outTopoModifier; ///< Pointer to the output topology modifier
};

} //namespace sofa::component::topology::mapping
