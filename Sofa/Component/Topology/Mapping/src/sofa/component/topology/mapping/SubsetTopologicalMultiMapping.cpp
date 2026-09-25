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
#include <sofa/component/topology/mapping/SubsetTopologicalMultiMapping.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/accessor.h>

namespace sofa::component::topology::mapping
{

void registerSubsetTopologicalMultiMapping(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(
        core::ObjectRegistrationData("Merges multiple input topologies (points, edges, triangles, "
                                     "quads, tetrahedra, hexahedra) "
                                     "into a single output topology with index remapping. "
                                     "Exposes indexPairs Data for linking to SubsetMultiMapping.")
            .add<SubsetTopologicalMultiMapping>());
}

SubsetTopologicalMultiMapping::SubsetTopologicalMultiMapping()
    : l_inputs(initLink("input", "Input topology sources to merge")),
      l_output(initLink("output", "Output merged topology")),
      d_flipNormals(
          initData(&d_flipNormals, sofa::type::vector<bool>(), "flipNormals",
                   "Per-source boolean flags to reverse triangle and quad winding order")),
      d_indexPairs(initData(&d_indexPairs, sofa::type::vector<unsigned>(), "indexPairs",
                            "Output: flat array of (source_index, coord_in_source) pairs"))
{
    d_indexPairs.setReadOnly(true);
}

SubsetTopologicalMultiMapping::~SubsetTopologicalMultiMapping() = default;

void SubsetTopologicalMultiMapping::init()
{
    BaseObject::init();

    if (l_inputs.size())
    {
        for (const auto inputTopology : l_inputs)
        {
            if (!inputTopology.get())
            {
                msg_error() << "Input topology '" << inputTopology.path << "' could not be found.";
                d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
                return;
            }
        }
    }
    else
    {
        msg_error() << "No input topologies found to be linked. Set the 'input' Data with "
                       "paths to the topologies considered as inputs for the mapping.";
        d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    if (!l_output.get())
    {
        msg_error()
            << "Null output topology. Set the 'output' Data with the path to the output topology.";
        d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    mapTopologies();

    d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

void SubsetTopologicalMultiMapping::reinit()
{
    if (d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Invalid) return;

    mapTopologies();
}

void SubsetTopologicalMultiMapping::mapTopologies()
{
    const auto numInputs = l_inputs.size();

    l_output->clear();
    m_pointOffsets.clear();

    // Compute per-source point offsets
    sofa::Size totalPoints{0};
    for (const auto input : l_inputs)
    {
        m_pointOffsets.push_back(totalPoints);
        totalPoints += input->getNbPoints();
    }
    l_output->setNbPoints(totalPoints);

    // Build indexPairs necessary to attach mechanical mapping through SubsetMultiMapping link
    {
        auto indexPairs = sofa::helper::getWriteOnlyAccessor(d_indexPairs);
        indexPairs.clear();
        indexPairs.reserve(totalPoints * 2);

        for (sofa::Size srcIdx = 0; srcIdx < numInputs; ++srcIdx)
        {
            const auto nbPts = l_inputs.get(srcIdx)->getNbPoints();
            for (sofa::Size p = 0; p < nbPts; ++p)
            {
                indexPairs.push_back(srcIdx);
                indexPairs.push_back(p);
            }
        }
    }

    // Pre-normalize flipNormals to exactly numInputs entries
    auto flipVec = sofa::helper::getWriteAccessor(d_flipNormals);
    if (flipVec.size() > numInputs)
    {
        msg_warning() << "flipNormals has " << flipVec.size() << " entries but there are only "
                      << numInputs << " input topologies. Extra entries will be discarded.";
        flipVec.resize(numInputs);
    }
    else if (flipVec.size() < numInputs)
    {
        flipVec.resize(numInputs, false);
    }

    // Concatenate edges from sources with offset remapping
    for (sofa::Size srcIdx = 0; srcIdx < numInputs; ++srcIdx)
    {
        const sofa::Size offset = m_pointOffsets[srcIdx];

        for (const auto& edge : l_inputs.get(srcIdx)->getEdges())
            l_output->addEdge(edge[0] + offset, edge[1] + offset);
    }

    // Concatenate triangles with offset remapping; optionally flip normals
    for (sofa::Size srcIdx = 0; srcIdx < numInputs; ++srcIdx)
    {
        const sofa::Size offset = m_pointOffsets[srcIdx];
        const bool flip = flipVec[srcIdx];

        for (const auto& tri : l_inputs.get(srcIdx)->getTriangles())
        {
            if (flip)
                l_output->addTriangle(tri[0] + offset, tri[2] + offset, tri[1] + offset);
            else
                l_output->addTriangle(tri[0] + offset, tri[1] + offset, tri[2] + offset);
        }
    }

    // Concatenate quads with offset remapping; optionally flip normals
    for (sofa::Size srcIdx = 0; srcIdx < numInputs; ++srcIdx)
    {
        const sofa::Size offset = m_pointOffsets[srcIdx];
        const bool flip = flipVec[srcIdx];

        for (auto& quad : l_inputs.get(srcIdx)->getQuads())
        {
            if (flip)
                l_output->addQuad(quad[0] + offset, quad[3] + offset, quad[2] + offset,
                                  quad[1] + offset);
            else
                l_output->addQuad(quad[0] + offset, quad[1] + offset, quad[2] + offset,
                                  quad[3] + offset);
        }
    }

    // Concatenate tetrahedra with offset remapping
    for (sofa::Size srcIdx = 0; srcIdx < numInputs; ++srcIdx)
    {
        const sofa::Size offset = m_pointOffsets[srcIdx];

        for (const auto& tet : l_inputs.get(srcIdx)->getTetrahedra())
            l_output->addTetra(tet[0] + offset, tet[1] + offset, tet[2] + offset, tet[3] + offset);
    }

    // Concatenate hexahedra with offset remapping
    for (sofa::Size srcIdx = 0; srcIdx < numInputs; ++srcIdx)
    {
        const sofa::Size offset = m_pointOffsets[srcIdx];

        for (const auto& hex : l_inputs.get(srcIdx)->getHexahedra())
        {
            l_output->addHexa(hex[0] + offset, hex[1] + offset, hex[2] + offset, hex[3] + offset,
                              hex[4] + offset, hex[5] + offset, hex[6] + offset, hex[7] + offset);
        }
    }

    msg_info() << "Merged " << numInputs << " topologies: " << totalPoints << " points, "
               << l_output->getNbEdges() << " edges, " << l_output->getNbTriangles()
               << " triangles, " << l_output->getNbQuads() << " quads, "
               << l_output->getNbTetrahedra() << " tetrahedra, " << l_output->getNbHexahedra()
               << " hexahedra.";
}

}  // namespace sofa::component::topology::mapping
