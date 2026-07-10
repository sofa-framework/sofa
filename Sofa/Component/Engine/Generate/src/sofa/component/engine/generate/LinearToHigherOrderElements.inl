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
#include <sofa/component/engine/generate/LinearToHigherOrderElements.h>
#include <sofa/core/behavior/SingleStateAccessor.h>

namespace sofa::component::engine::generate
{

template <class DataTypes>
LinearToHigherOrderElements<DataTypes>::LinearToHigherOrderElements()
    : d_position(initData(&d_position, sofa::VecCoord_t<DataTypes>{}, "position", "Output position"))
    , d_quadraticEdges(initData(&d_quadraticEdges, SeqElement<sofa::geometry::QuadraticEdge>{},
        "quadratic_edges", "List of quadratic edges"))
    , d_quadraticTriangles(initData(&d_quadraticTriangles, SeqElement<sofa::geometry::QuadraticTriangle>{},
        "quadratic_triangles", "List of quadratic triangles"))
    , d_quadraticTetrahedra(initData(&d_quadraticTetrahedra, SeqElement<sofa::geometry::QuadraticTetrahedron>{},
        "quadratic_tetrahedra", "List of quadratic tetrahedra"))
{
    this->addOutput(&d_position);
    this->addOutput(&d_quadraticEdges);
    this->addOutput(&d_quadraticTriangles);
    this->addOutput(&d_quadraticTetrahedra);
}

template <class DataTypes>
void LinearToHigherOrderElements<DataTypes>::init()
{
    DataEngine::init();

    if (!this->isComponentStateInvalid())
    {
        this->validateTopology();
    }

    if (!this->isComponentStateInvalid())
    {
        sofa::core::behavior::SingleStateAccessor<DataTypes>::init();
    }
}


struct TetrahedronEdge
{
    TetrahedronEdge(sofa::Index p, sofa::Index q) : a(std::min(p, q)), b(std::max(p, q)) {}
    sofa::Index a, b;
    bool operator==(const TetrahedronEdge& e) const
    {
        return a == e.a && b == e.b;
    }
};


template <class DataTypes>
void LinearToHigherOrderElements<DataTypes>::doUpdate()
{
    if (!l_topology)
        return;

    //input elements
    const auto& edges = l_topology->getElements<sofa::geometry::Edge>();
    const auto& triangles = l_topology->getElements<sofa::geometry::Triangle>();
    const auto& tetrahedra = l_topology->getElements<sofa::geometry::Tetrahedron>();

    //input position
    const auto inPosition = this->mstate->readPositions();

    //output elements
    auto quadraticEdges = sofa::helper::getWriteOnlyAccessor(d_quadraticEdges);
    auto quadraticTriangles = sofa::helper::getWriteOnlyAccessor(d_quadraticTriangles);
    auto quadraticTetrahedra = sofa::helper::getWriteOnlyAccessor(d_quadraticTetrahedra);

    //output position
    auto outPosition = sofa::helper::getWriteOnlyAccessor(d_position);
    outPosition.wref() = inPosition.ref();

    auto hash = [&inPosition](const TetrahedronEdge& edge){ return edge.a + inPosition.size() * edge.b; };
    std::unordered_map<TetrahedronEdge, sofa::Size, decltype(hash)> newPointsMap(10, hash);

    auto getOrAddMidPoint = [&](sofa::Index a, sofa::Index b)
    {
        TetrahedronEdge edge{a, b};
        const auto [it, success] = newPointsMap.insert(std::make_pair(edge, outPosition.size()));
        if (success)
        {
            outPosition.push_back(0.5 * (inPosition[a] + inPosition[b]));
        }
        return it->second;
    };

    for (const auto& element : edges)
    {
        quadraticEdges.emplace_back(element[0], element[1], getOrAddMidPoint(element[0], element[1]));
    }

    for (const auto& element : triangles)
    {
        quadraticTriangles.emplace_back(element[0], element[1], element[2],
            getOrAddMidPoint(element[0], element[1]),
            getOrAddMidPoint(element[0], element[2]),
            getOrAddMidPoint(element[1], element[2]));
    }

    for (const auto& element : tetrahedra)
    {
        std::array<sofa::Index, 6> newIndices;
        static constexpr std::array<std::pair<sofa::Index, sofa::Index>, 6> listEdgesInTetra {{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
        for (std::size_t i = 0; i < listEdgesInTetra.size(); ++i)
        {
            const auto [a, b] = listEdgesInTetra[i];
            newIndices[i] = getOrAddMidPoint(element[a], element[b]);
        }

        quadraticTetrahedra.emplace_back(element[0], element[1], element[2], element[3],
            newIndices[0], newIndices[1], newIndices[2], newIndices[3],
            newIndices[4], newIndices[5]);
    }
}

}  // namespace sofa::component::engine::generate
