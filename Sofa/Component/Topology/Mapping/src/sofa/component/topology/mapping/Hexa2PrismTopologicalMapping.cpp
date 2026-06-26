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
#include <sofa/component/topology/mapping/Hexa2PrismTopologicalMapping.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::topology::mapping
{

void registerHexa2PrismTopologicalMapping(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Topological mapping where HexahedronSetTopology is converted to PrismSetTopology")
        .add< Hexa2PrismTopologicalMapping >());
}

Hexa2PrismTopologicalMapping::Hexa2PrismTopologicalMapping()
{
    m_inputType = geometry::ElementType::HEXAHEDRON;
    m_outputType = geometry::ElementType::PRISM;
}

void Hexa2PrismTopologicalMapping::init()
{
    Inherit1::init();

    if (toModel == nullptr)
    {
        msg_error() << "No target topology container found.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    convertHexaToPrisms();
}

void Hexa2PrismTopologicalMapping::convertHexaToPrisms()
{
    // Clear output topology
    toModel->clear();

    // Set the same number of points
    toModel->setNbPoints(fromModel->getNbPoints());

    auto Loc2GlobVec = sofa::helper::getWriteOnlyAccessor(Loc2GlobDataVec);
    Loc2GlobVec.clear();
    Glob2LocMap.clear();

    const sofa::Size nbCubes = fromModel->getNbHexahedra();

    static constexpr std::size_t numberPrismsInHexa = 2;
    Loc2GlobVec.reserve(nbCubes * numberPrismsInHexa);

    // Tessellation of each cube into 2 triangular prisms
    // Hexahedron vertices:
    //     Y  n3---------n2
    //     ^  /          /|
    //     | /          / |
    //     n7---------n6  |
    //     |          |   |
    //     |  n0------|--n1
    //     | /        | /
    //     |/         |/
    //     n4---------n5-->X
    //    /
    //   /
    //  Z
    //
    // Decomposition into 2 prisms:
    // - Prism 1: vertices [0, 5, 1] (bottom triangle) and [3, 6, 2] (top triangle)
    // - Prism 2: vertices [0, 4, 5] (bottom triangle) and [3, 7, 6] (top triangle)

    for (size_t i = 0; i < nbCubes; ++i)
    {
        core::topology::BaseMeshTopology::Hexa c = fromModel->getHexahedron(i);

        // Standard decomposition ensuring face consistency between neighbors
        toModel->addPrism(c[0], c[5], c[1], c[3], c[6], c[2]);  // Prism 1
        toModel->addPrism(c[0], c[4], c[5], c[3], c[7], c[6]);  // Prism 2

        for (unsigned j = 0; j < numberPrismsInHexa; ++j)
        {
            Loc2GlobVec.push_back(i);
        }
        Glob2LocMap[i] = static_cast<unsigned int>(Loc2GlobVec.size()) - 1;
    }

    // Need to fully init the target topology
    toModel->init();

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

Index Hexa2PrismTopologicalMapping::getFromIndex(Index /*ind*/)
{
    return sofa::InvalidID;
}

void Hexa2PrismTopologicalMapping::updateTopologicalMappingTopDown()
{
    msg_warning() << "Method Hexa2PrismTopologicalMapping::updateTopologicalMappingTopDown() not yet implemented!";
    // TODO...
}

} // namespace sofa::component::topology::mapping
