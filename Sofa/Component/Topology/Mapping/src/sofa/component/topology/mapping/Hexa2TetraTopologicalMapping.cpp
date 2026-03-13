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
#include <sofa/component/topology/mapping/Hexa2TetraTopologicalMapping.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyModifier.h>

#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyModifier.h>

#include <sofa/core/topology/TopologyChange.h>

#include <sofa/component/topology/container/grid/GridTopology.h>

#include <sofa/type/Vec.h>
#include <map>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::topology::mapping
{

using namespace sofa::component::topology::mapping;
using namespace sofa::core::topology;

void registerHexa2TetraTopologicalMapping(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Topological mapping where HexahedronSetTopology is converted to TetrahedronSetTopology")
        .add< Hexa2TetraTopologicalMapping >());
}

Hexa2TetraTopologicalMapping::Hexa2TetraTopologicalMapping()
    : sofa::core::topology::TopologicalMapping(),
      d_swapping(initData(&d_swapping, false, "swapping",
                          "Boolean enabling to swap edges to hexahedrons based on their grid "
                          "position in order to avoid numerical bias effect"))
{
    m_inputType = geometry::ElementType::HEXAHEDRON;
    m_outputType = geometry::ElementType::TETRAHEDRON;
}

void Hexa2TetraTopologicalMapping::init()
{
    using namespace container::dynamic;

    Inherit1::init();

    if (toModel == nullptr)
    {
        msg_error() << "No target topology container found.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    // INITIALISATION of TETRAHEDRAL mesh from HEXAHEDRAL mesh :

    // Clear output topology
    toModel->clear();

    // Set the same number of points
    toModel->setNbPoints(fromModel->getNbPoints());

    auto Loc2GlobVec = sofa::helper::getWriteOnlyAccessor(Loc2GlobDataVec);
    Loc2GlobVec.clear();
    Glob2LocMap.clear();

    const sofa::Size nbCubes = fromModel->getNbHexahedra();

    // These values are only correct if the mesh is a grid topology
    int nx = 2;
    int ny = 1;
    // int nz = 1;
    {
        const auto* grid = dynamic_cast<container::grid::GridTopology*>(fromModel.get());
        if (grid != nullptr)
        {
            nx = grid->getNx() - 1;
            ny = grid->getNy() - 1;
            // nz = grid->getNz()-1;
        }
    }

    static constexpr int numberTetraInHexa = 6;
    Loc2GlobVec.reserve(nbCubes * numberTetraInHexa);

    const bool swapping = d_swapping.getValue();

    // Tessellation of each cube into 6 tetrahedra
    for (sofa::Index i = 0; i < nbCubes; ++i)
    {
        // take a copy of the hexahedron in case vertices must be swapped
        sofa::topology::Hexahedron c = fromModel->getHexahedron(i);

        bool swapped = false;

        if (swapping)
        {
            // Check if hexahedron is at an even x-position in the grid
            if (!((i % nx) & 1))
            {
                // swap all nodes on the X edges
                //     Y  n3---------n2             Y  n2---------n3
                //     ^  /          /|             ^  /          /|
                //     | /          / |             | /          / |
                //     n7---------n6  |             n6---------n7  |
                //     |          |   |             |          |   |
                //     |  n0------|--n1      =>     |  n1------|--n0
                //     | /        | /               | /        | /
                //     |/         |/                |/         |/
                //     n4---------n5-->X            n5---------n4-->X
                //    /                            /
                //   /                            /
                //  Z                            Z
                for (const auto [v0, v1] : sofa::geometry::Hexahedron::xEdges)
                {
                    std::swap(c[v0], c[v1]);
                }
                swapped = !swapped;
            }
            // Check if hexahedron is at an odd y-position in the grid
            if (((i / nx) % ny) & 1)
            {
                // swap all nodes on the Y edges
                //     Y  n3---------n2             Y  n0---------n1
                //     ^  /          /|             ^  /          /|
                //     | /          / |             | /          / |
                //     n7---------n6  |             n4---------n5  |
                //     |          |   |             |          |   |
                //     |  n0------|--n1      =>     |  n3------|--n2
                //     | /        | /               | /        | /
                //     |/         |/                |/         |/
                //     n4---------n5-->X            n7---------n6-->X
                //    /                            /
                //   /                            /
                //  Z                            Z
                for (const auto [v0, v1] : sofa::geometry::Hexahedron::yEdges)
                {
                    std::swap(c[v0], c[v1]);
                }
                swapped = !swapped;
            }
            // Check if hexahedron is at an odd z-position in the grid
            if ((i / (nx * ny)) & 1)
            {
                // swap all nodes on the Z edges
                //     Y  n3---------n2             Y  n7---------n6
                //     ^  /          /|             ^  /          /|
                //     | /          / |             | /          / |
                //     n7---------n6  |             n3---------n2  |
                //     |          |   |             |          |   |
                //     |  n0------|--n1      =>     |  n4------|--n5
                //     | /        | /               | /        | /
                //     |/         |/                |/         |/
                //     n4---------n5-->X            n0---------n1-->X
                //    /                            /
                //   /                            /
                //  Z                            Z
                for (const auto [v0, v1] : sofa::geometry::Hexahedron::zEdges)
                {
                    std::swap(c[v0], c[v1]);
                }
                swapped = !swapped;
            }
        }

        static constexpr std::array<std::array<sofa::Index, 4>, 6> nonSwappedPattern{
            {{0, 5, 1, 6}, {0, 1, 3, 6}, {1, 3, 6, 2}, {6, 3, 0, 7}, {6, 7, 0, 5}, {7, 5, 4, 0}}};

        static constexpr std::array<std::array<sofa::Index, 4>, 6> swappedPattern{
            {{0, 5, 6, 1}, {0, 1, 6, 3}, {1, 3, 2, 6}, {6, 3, 7, 0}, {6, 7, 5, 0}, {7, 5, 0, 4}}};

        const auto& pattern = swapped ? swappedPattern : nonSwappedPattern;
        for (const auto& id : pattern)
        {
            toModel->addTetra(c[id[0]], c[id[1]], c[id[2]], c[id[3]]);
        }

        for (int j = 0; j < numberTetraInHexa; j++)
        {
            Loc2GlobVec.push_back(i);
        }
        Glob2LocMap[i] = static_cast<unsigned int>(Loc2GlobVec.size()) - 1;
    }

    // Need to fully init the target topology
    toModel->init();

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

void Hexa2TetraTopologicalMapping::updateTopologicalMappingTopDown()
{
    msg_warning() << "Method Hexa2TetraTopologicalMapping::updateTopologicalMappingTopDown() not "
                     "yet implemented!";
    // TODO...
}

Index Hexa2TetraTopologicalMapping::getFromIndex(Index /*ind*/) { return sofa::InvalidID; }


} //namespace sofa::component::topology::mapping
