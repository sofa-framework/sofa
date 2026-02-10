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
    factory->registerObjects(core::ObjectRegistrationData(
        "Merges multiple input topologies (points, edges, triangles, tetrahedra) into a single "
        "output topology with index remapping. Optionally populates indexPairs for "
        "SubsetMultiMapping coordination.")
        .add<SubsetTopologicalMultiMapping>());
}

SubsetTopologicalMultiMapping::SubsetTopologicalMultiMapping()
    : l_inputTopologies(initLink("input", "Input topology sources to merge"))
    , l_outputTopology(initLink("output", "Output merged topology"))
    , l_subsetMultiMapping(initLink("subsetMultiMapping",
          "Optional link to a SubsetMultiMapping to auto-populate its indexPairs"))
    , d_flipNormals(initData(&d_flipNormals, sofa::type::vector<bool>(),
          "flipNormals",
          "Per-source boolean flags to reverse triangle winding order"))
    , d_indexPairs(initData(&d_indexPairs, sofa::type::vector<unsigned>(),
          "indexPairs",
          "Output: flat array of (source_index, coord_in_source) pairs"))
{
    d_indexPairs.setReadOnly(true);
}

SubsetTopologicalMultiMapping::~SubsetTopologicalMultiMapping() = default;

void SubsetTopologicalMultiMapping::init()
{
    BaseObject::init();

    const auto numInputs = l_inputTopologies.size();
    if (numInputs == 0)
    {
        msg_error() << "No input topologies linked. Set the 'input' attribute with "
                        "space-separated paths to input topologies.";
        d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    for (std::size_t i = 0; i < numInputs; ++i)
    {
        if (l_inputTopologies.get(i) == nullptr)
        {
            msg_error() << "Input topology #" << i << " is null.";
            d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            return;
        }
    }

    if (l_outputTopology.get() == nullptr)
    {
        msg_error() << "Output topology is null. Set the 'output' attribute.";
        d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    doMerge();

    if (l_subsetMultiMapping.get() != nullptr)
    {
        populateSubsetMultiMappingIndexPairs();
    }

    d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

void SubsetTopologicalMultiMapping::reinit()
{
    if (d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Invalid)
        return;

    doMerge();

    if (l_subsetMultiMapping.get() != nullptr)
    {
        populateSubsetMultiMappingIndexPairs();
    }
}

void SubsetTopologicalMultiMapping::doMerge()
{
    BaseMeshTopology* output = l_outputTopology.get();
    const auto numInputs = l_inputTopologies.size();

    // Phase 1: Clear output topology
    output->clear();

    // Phase 2: Compute per-source point offsets
    m_pointOffsets.resize(numInputs);
    Index totalPoints = 0;
    for (std::size_t i = 0; i < numInputs; ++i)
    {
        m_pointOffsets[i] = totalPoints;
        totalPoints += static_cast<Index>(l_inputTopologies.get(i)->getNbPoints());
    }

    // Phase 3: Set total point count on output
    output->setNbPoints(static_cast<sofa::Size>(totalPoints));

    // Phase 4: Build indexPairs for SubsetMultiMapping coordination
    {
        auto indexPairs = sofa::helper::getWriteOnlyAccessor(d_indexPairs);
        indexPairs.clear();
        indexPairs.reserve(static_cast<std::size_t>(totalPoints) * 2);

        for (std::size_t srcIdx = 0; srcIdx < numInputs; ++srcIdx)
        {
            const auto nbPts = static_cast<Index>(l_inputTopologies.get(srcIdx)->getNbPoints());
            for (Index p = 0; p < nbPts; ++p)
            {
                indexPairs.push_back(static_cast<unsigned>(srcIdx));
                indexPairs.push_back(static_cast<unsigned>(p));
            }
        }
    }

    // Phase 5: Read flip flags
    const auto& flipVec = d_flipNormals.getValue();
    if (!flipVec.empty() && flipVec.size() != numInputs)
    {
        msg_warning() << "flipNormals has " << flipVec.size() << " entries but there are "
                      << numInputs << " input topologies. Missing entries default to false.";
    }

    // Phase 6: Concatenate edges with offset remapping
    for (std::size_t srcIdx = 0; srcIdx < numInputs; ++srcIdx)
    {
        BaseMeshTopology* input = l_inputTopologies.get(srcIdx);
        const Index offset = m_pointOffsets[srcIdx];
        const auto& edges = input->getEdges();

        for (std::size_t e = 0; e < edges.size(); ++e)
        {
            output->addEdge(edges[e][0] + offset, edges[e][1] + offset);
        }
    }

    // Phase 7: Concatenate triangles with offset remapping + optional flip
    for (std::size_t srcIdx = 0; srcIdx < numInputs; ++srcIdx)
    {
        BaseMeshTopology* input = l_inputTopologies.get(srcIdx);
        const Index offset = m_pointOffsets[srcIdx];
        const bool flip = (srcIdx < flipVec.size()) ? flipVec[srcIdx] : false;
        const auto& triangles = input->getTriangles();

        for (std::size_t t = 0; t < triangles.size(); ++t)
        {
            const auto& tri = triangles[t];
            if (flip)
                output->addTriangle(tri[0] + offset, tri[2] + offset, tri[1] + offset);
            else
                output->addTriangle(tri[0] + offset, tri[1] + offset, tri[2] + offset);
        }
    }

    // Phase 8: Concatenate tetrahedra with offset remapping
    for (std::size_t srcIdx = 0; srcIdx < numInputs; ++srcIdx)
    {
        BaseMeshTopology* input = l_inputTopologies.get(srcIdx);
        const Index offset = m_pointOffsets[srcIdx];
        const auto& tetrahedra = input->getTetrahedra();

        for (std::size_t t = 0; t < tetrahedra.size(); ++t)
        {
            const auto& tet = tetrahedra[t];
            output->addTetra(tet[0] + offset, tet[1] + offset,
                             tet[2] + offset, tet[3] + offset);
        }
    }

    msg_info() << "Merged " << numInputs << " topologies: "
               << totalPoints << " points, "
               << output->getNbEdges() << " edges, "
               << output->getNbTriangles() << " triangles, "
               << output->getNbTetrahedra() << " tetrahedra.";
}

void SubsetTopologicalMultiMapping::populateSubsetMultiMappingIndexPairs()
{
    auto* targetObj = l_subsetMultiMapping.get();
    if (!targetObj)
        return;

    auto* targetData = targetObj->findData("indexPairs");
    if (!targetData)
    {
        msg_error() << "Linked object '" << targetObj->getName()
                    << "' has no 'indexPairs' Data field. "
                       "Ensure it is a SubsetMultiMapping.";
        return;
    }

    targetData->copyValueFrom(&d_indexPairs);

    msg_info() << "Set " << (d_indexPairs.getValue().size() / 2)
               << " indexPairs on '" << targetObj->getName() << "'.";
}

} // namespace sofa::component::topology::mapping
