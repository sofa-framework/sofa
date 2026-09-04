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
    , d_quadraticQuads(initData(&d_quadraticQuads, SeqElement<sofa::geometry::QuadraticQuad>{},
        "quadratic_quads", "List of quadratic quads"))
    , d_quadraticTetrahedra(initData(&d_quadraticTetrahedra, SeqElement<sofa::geometry::QuadraticTetrahedron>{},
        "quadratic_tetrahedra", "List of quadratic tetrahedra"))
    , d_quadraticHexahedra(initData(&d_quadraticHexahedra, SeqElement<sofa::geometry::QuadraticHexahedron>{},
        "quadratic_hexahedra", "List of quadratic hexahedra"))
    , d_computeFromEdges(initData(&d_computeFromEdges, true, "computeFromEdges",
        "Use edges from the input to generate higher-order edges"))
    , d_computeFromTriangles(initData(&d_computeFromTriangles, true, "computeFromTriangles",
            "Use triangles from the input to generate higher-order triangles"))
    , d_computeFromQuads(initData(&d_computeFromQuads, true, "computeFromQuads",
            "Use quads from the input to generate higher-order quads"))
    , d_computeFromTetrahedra(initData(&d_computeFromTetrahedra, true, "computeFromTetrahedra",
            "Use tetrahedra from the input to generate higher-order tetrahedra"))
    , d_computeFromHexahedra(initData(&d_computeFromHexahedra, true, "computeFromHexahedra",
            "Use hexahedra from the input to generate higher-order hexahedra"))
{
    this->addOutput(&d_position);
    this->addOutput(&d_quadraticEdges);
    this->addOutput(&d_quadraticTriangles);
    this->addOutput(&d_quadraticQuads);
    this->addOutput(&d_quadraticTetrahedra);
    this->addOutput(&d_quadraticHexahedra);
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

struct Face {
    std::array<sofa::Index, 4> indices;
    Face(sofa::Index a, sofa::Index b, sofa::Index c, sofa::Index d) {
        indices = {a, b, c, d};
        std::sort(indices.begin(), indices.end());
    }
    bool operator==(const Face& other) const { return indices == other.indices; }
};


template <class DataTypes>
void LinearToHigherOrderElements<DataTypes>::doUpdate()
{
    if (!l_topology)
        return;

    //input elements
    const auto& edges = l_topology->getElements<sofa::geometry::Edge>();
    const auto& triangles = l_topology->getElements<sofa::geometry::Triangle>();
    const auto& quads = l_topology->getElements<sofa::geometry::Quad>();
    const auto& tetrahedra = l_topology->getElements<sofa::geometry::Tetrahedron>();
    const auto& hexahedra = l_topology->getElements<sofa::geometry::Hexahedron>();

    //input position
    const auto inPosition = this->mstate->readPositions();

    //output elements
    auto quadraticEdges = sofa::helper::getWriteOnlyAccessor(d_quadraticEdges);
    auto quadraticTriangles = sofa::helper::getWriteOnlyAccessor(d_quadraticTriangles);
    auto quadraticQuads = sofa::helper::getWriteOnlyAccessor(d_quadraticQuads);
    auto quadraticTetrahedra = sofa::helper::getWriteOnlyAccessor(d_quadraticTetrahedra);
    auto quadraticHexahedra = sofa::helper::getWriteOnlyAccessor(d_quadraticHexahedra);

    //output position
    auto outPosition = sofa::helper::getWriteOnlyAccessor(d_position);
    outPosition.wref() = inPosition.ref();

    auto edgeHash = [&inPosition](const TetrahedronEdge& edge){ return edge.a + inPosition.size() * edge.b; };
    std::unordered_map<TetrahedronEdge, sofa::Size, decltype(edgeHash)> midEdgePointsMap(10, edgeHash);

    auto getOrAddMidEdgePoint = [&](sofa::Index a, sofa::Index b)
    {
        TetrahedronEdge edge{a, b};
        const auto [it, success] = midEdgePointsMap.insert(std::make_pair(edge, outPosition.size()));
        if (success)
        {
            outPosition.push_back(0.5 * (inPosition[a] + inPosition[b]));
        }
        return it->second;
    };

    auto faceHash = [](const Face& f) {
        size_t h = 0;
        for (auto i : f.indices) h ^= std::hash<sofa::Index>{}(i) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    };
    std::unordered_map<Face, sofa::Size, decltype(faceHash)> midFacePointsMap(10, faceHash);

    auto getOrAddMidFacePoint = [&](sofa::Index a, sofa::Index b, sofa::Index c, sofa::Index d)
    {
        Face face{a, b, c, d};
        const auto [it, success] = midFacePointsMap.insert(std::make_pair(face, outPosition.size()));
        if (success)
        {
            outPosition.push_back(0.25 * (inPosition[a] + inPosition[b] + inPosition[c] + inPosition[d]));
        }
        return it->second;
    };

    if (d_computeFromEdges.getValue())
    for (const auto& element : edges)
    {
        quadraticEdges.emplace_back(element[0], element[1], getOrAddMidEdgePoint(element[0], element[1]));
    }

    if (d_computeFromTriangles.getValue())
    for (const auto& element : triangles)
    {
        quadraticTriangles.emplace_back(element[0], element[1], element[2],
            getOrAddMidEdgePoint(element[0], element[1]),
            getOrAddMidEdgePoint(element[1], element[2]),
            getOrAddMidEdgePoint(element[2], element[0]));
    }

    if (d_computeFromQuads.getValue())
    for (const auto& element : quads)
    {
        quadraticQuads.emplace_back(element[0], element[1], element[2], element[3],
            getOrAddMidEdgePoint(element[0], element[1]),
            getOrAddMidEdgePoint(element[1], element[2]),
            getOrAddMidEdgePoint(element[2], element[3]),
            getOrAddMidEdgePoint(element[3], element[0]),
            getOrAddMidFacePoint(element[0], element[1], element[2], element[3]));
    }

    if (d_computeFromTetrahedra.getValue())
    for (const auto& element : tetrahedra)
    {
        std::array<sofa::Index, 6> newIndices;
        for (std::size_t i = 0; i < 6; ++i)
        {
            const auto [a, b] = core::topology::edgesInTetrahedronArray[i];
            newIndices[i] = getOrAddMidEdgePoint(element[a], element[b]);
        }

        quadraticTetrahedra.emplace_back(element[0], element[1], element[2], element[3],
            newIndices[0], newIndices[1], newIndices[2], newIndices[3],
            newIndices[4], newIndices[5]);
    }

    if (d_computeFromHexahedra.getValue())
    for (const auto& element : hexahedra)
    {
        std::array<sofa::Index, 12> midEdges;
        for (std::size_t i = 0; i < 12; ++i)
        {
            midEdges[i] =
               getOrAddMidEdgePoint(element[core::topology::edgesInHexahedronArray[i][0]],
                                    element[core::topology::edgesInHexahedronArray[i][1]]);
        }

        std::array<sofa::Index, 6> midFaces;
        for (std::size_t i = 0; i < 6; ++i)
        {
            const auto quad = core::topology::quadsOrientationInHexahedronArray[i];
            midFaces[i] = getOrAddMidFacePoint(element[quad[0]], element[quad[1]], element[quad[2]], element[quad[3]]);
        }

        sofa::Index midVolume = outPosition.size();
        outPosition.push_back(0.125 * (inPosition[element[0]] + inPosition[element[1]] + inPosition[element[2]] + inPosition[element[3]] +
                                       inPosition[element[4]] + inPosition[element[5]] + inPosition[element[6]] + inPosition[element[7]]));

        quadraticHexahedra.emplace_back(element[0], element[1], element[2], element[3], element[4], element[5], element[6], element[7],
            midEdges[0], midEdges[1], midEdges[2], midEdges[3], midEdges[4], midEdges[5], midEdges[6], midEdges[7], midEdges[8], midEdges[9], midEdges[10], midEdges[11],
            midFaces[0], midFaces[1], midFaces[2], midFaces[3], midFaces[4], midFaces[5], midVolume);
    }
}

}  // namespace sofa::component::engine::generate
