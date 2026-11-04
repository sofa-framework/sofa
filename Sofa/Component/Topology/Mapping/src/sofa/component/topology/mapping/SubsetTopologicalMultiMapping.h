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
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa::component::topology::mapping
{

/**
 * @class SubsetTopologicalMultiMapping
 * @brief Merges multiple input topologies into a single output topology.
 *
 * For each input topology, points are concatenated with index offsets, and geometric primitives
 * (edges/triangles/quads/tetrahedra/hexahedra are remapped accordingly. Triangle/quad winding
 * order can be reversed per source via flipNormals.
 *
 * This is the topological counterpart to SubsetMultiMapping.
 * The indexPairs output can be linked to a SubsetMultiMapping in scenes.
 *
 * Merging is performed once during init(). Dynamic topology changes are not supported.
 */
class SOFA_COMPONENT_TOPOLOGY_MAPPING_API SubsetTopologicalMultiMapping
    : public sofa::core::objectmodel::BaseObject
{
   public:
    SOFA_CLASS(SubsetTopologicalMultiMapping, sofa::core::objectmodel::BaseObject);

    using Index = sofa::Index;
    using BaseMeshTopology = sofa::core::topology::BaseMeshTopology;

    void init() override;
    void reinit() override;

    /// N input topologies
    MultiLink<SubsetTopologicalMultiMapping, BaseMeshTopology,
              sofa::core::objectmodel::BaseLink::FLAG_STOREPATH |
                  sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK>
        l_inputs;

    /// Single output topology
    SingleLink<SubsetTopologicalMultiMapping, BaseMeshTopology,
               sofa::core::objectmodel::BaseLink::FLAG_STOREPATH |
                   sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK>
        l_output;

    Data<sofa::type::vector<bool>>
        d_flipNormals;  ///< Per-source flags to reverse triangle and/or quad winding order
    Data<sofa::type::vector<unsigned>>
        d_indexPairs;  ///< Output: flat array of (source_index, coord_in_source) pairs,
                       ///< linkable to SubsetMultiMapping

   protected:
    SubsetTopologicalMultiMapping();
    ~SubsetTopologicalMultiMapping() override;

   private:
    void mapTopologies();

    /// Per-source cumulative point offsets. m_pointOffsets[i] = total points from sources 0..i-1.
    sofa::type::vector<Index> m_pointOffsets;
};

}  // namespace sofa::component::topology::mapping
