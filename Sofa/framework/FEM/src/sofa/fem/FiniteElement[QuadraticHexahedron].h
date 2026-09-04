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
#include <sofa/fem/FiniteElement.h>

#if !defined(SOFA_FEM_FINITE_ELEMENT_QUADRATIC_HEXAHEDRON_CPP)
#include <sofa/defaulttype/VecTypes.h>
#endif

namespace sofa::fem
{

template <class DataTypes>
struct FiniteElement<sofa::geometry::QuadraticHexahedron, DataTypes>
{
    FINITEELEMENT_HEADER(sofa::geometry::QuadraticHexahedron, DataTypes, 3);
    static_assert(spatial_dimensions == 3, "Quadratic Hexahedrons are only defined in 3D");

    constexpr static std::array<ReferenceCoord, NumberOfNodesInElement> referenceElementNodes {{

        // 8 corner vertices
        /*00*/ {-1, -1, -1},  // vertex 0
        /*01*/ { 1, -1, -1},   // vertex 1
        /*02*/ { 1,  1, -1},    // vertex 2
        /*03*/ {-1,  1, -1},   // vertex 3
        /*04*/ {-1, -1,  1},   // vertex 4
        /*05*/ { 1, -1,  1},    // vertex 5
        /*06*/ { 1,  1,  1},     // vertex 6
        /*07*/ {-1,  1,  1},    // vertex 7

        // 12 mid-edge nodes
        /*08*/ { 0, -1, -1},   // mid-edge 0-1
        /*09*/ {-1,  0, -1},   // mid-edge 0-3
        /*10*/ {-1, -1,  0},   // mid-edge 0-4
        /*11*/ { 1,  0, -1},   // mid-edge 1-2
        /*12*/ { 1, -1,  0},   // mid-edge 1-5
        /*13*/ { 0,  1, -1},   // mid-edge 2-3
        /*14*/ { 1,  1,  0},   // mid-edge 2-6
        /*15*/ {-1,  1,  0},   // mid-edge 3-7
        /*16*/ { 0, -1,  1},   // mid-edge 4-5
        /*17*/ {-1,  0,  1},   // mid-edge 7-4
        /*18*/ { 1,  0,  1},   // mid-edge 5-6
        /*19*/ { 0,  1,  1},   // mid-edge 6-7

        // 6 face-center nodes
        /*20*/ { 0,  0, -1},    // face center bottom (0-1-2-3)
        /*21*/ { 0,  0,  1},     // face center top (4-5-6-7)
        /*22*/ { 0, -1,  0},    // face center front (0-1-5-4)
        /*23*/ { 1,  0,  0},     // face center right (1-2-6-5)
        /*24*/ { 0,  1,  0},     // face center back (2-3-7-6)
        /*25*/ {-1,  0,  0},    // face center left (3-0-4-7)

        // 1 volume center node
        /*26*/ {0, 0, 0}      // volume center
    }};

    static const sofa::type::vector<TopologyElement>& getElementSequence(sofa::core::topology::BaseMeshTopology& topology)
    {
        return topology.getElements<sofa::geometry::QuadraticHexahedron>();
    }

    static constexpr sofa::type::Vec<NumberOfNodesInElement, Real> shapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        const auto N0 = (q[0] - 1) * (q[1] - 1) * (q[2] - 1) * q[0] * q[1] * q[2] / 8;
        const auto N1 = (q[0] + 1) * (q[1] - 1) * (q[2] - 1) * q[0] * q[1] * q[2] / 8;
        const auto N2 = (q[0] + 1) * (q[1] + 1) * (q[2] - 1) * q[0] * q[1] * q[2] / 8;
        const auto N3 = (q[0] - 1) * (q[1] + 1) * (q[2] - 1) * q[0] * q[1] * q[2] / 8;
        const auto N4 = (q[0] - 1) * (q[1] - 1) * (q[2] + 1) * q[0] * q[1] * q[2] / 8;
        const auto N5 = (q[0] + 1) * (q[1] - 1) * (q[2] + 1) * q[0] * q[1] * q[2] / 8;
        const auto N6 = (q[0] + 1) * (q[1] + 1) * (q[2] + 1) * q[0] * q[1] * q[2] / 8;
        const auto N7 = (q[0] - 1) * (q[1] + 1) * (q[2] + 1) * q[0] * q[1] * q[2] / 8;
        const auto N8 = -(q[0] - 1) * (q[0] + 1) * (q[1] - 1) * (q[2] - 1) * q[1] * q[2] / 4;
        const auto N9 = -(q[0] - 1) * (q[1] - 1) * (q[1] + 1) * (q[2] - 1) * q[0] * q[2] / 4;
        const auto N10 = -(q[0] - 1) * (q[1] - 1) * (q[2] - 1) * (q[2] + 1) * q[0] * q[1] / 4;
        const auto N11 = -(q[0] + 1) * (q[1] - 1) * (q[1] + 1) * (q[2] - 1) * q[0] * q[2] / 4;
        const auto N12 = -(q[0] + 1) * (q[1] - 1) * (q[2] - 1) * (q[2] + 1) * q[0] * q[1] / 4;
        const auto N13 = -(q[0] - 1) * (q[0] + 1) * (q[1] + 1) * (q[2] - 1) * q[1] * q[2] / 4;
        const auto N14 = -(q[0] + 1) * (q[1] + 1) * (q[2] - 1) * (q[2] + 1) * q[0] * q[1] / 4;
        const auto N15 = -(q[0] - 1) * (q[1] + 1) * (q[2] - 1) * (q[2] + 1) * q[0] * q[1] / 4;
        const auto N16 = -(q[0] - 1) * (q[0] + 1) * (q[1] - 1) * (q[2] + 1) * q[1] * q[2] / 4;
        const auto N17 = -(q[0] - 1) * (q[1] - 1) * (q[1] + 1) * (q[2] + 1) * q[0] * q[2] / 4;
        const auto N18 = -(q[0] + 1) * (q[1] - 1) * (q[1] + 1) * (q[2] + 1) * q[0] * q[2] / 4;
        const auto N19 = -(q[0] - 1) * (q[0] + 1) * (q[1] + 1) * (q[2] + 1) * q[1] * q[2] / 4;
        const auto N20 = (q[0] - 1) * (q[0] + 1) * (q[1] - 1) * (q[1] + 1) * (q[2] - 1) * q[2] / 2;
        const auto N21 = (q[0] - 1) * (q[0] + 1) * (q[1] - 1) * (q[1] + 1) * (q[2] + 1) * q[2] / 2;
        const auto N22 = (q[0] - 1) * (q[0] + 1) * (q[1] - 1) * (q[2] - 1) * (q[2] + 1) * q[1] / 2;
        const auto N23 = (q[0] + 1) * (q[1] - 1) * (q[1] + 1) * (q[2] - 1) * (q[2] + 1) * q[0] / 2;
        const auto N24 = (q[0] - 1) * (q[0] + 1) * (q[1] + 1) * (q[2] - 1) * (q[2] + 1) * q[1] / 2;
        const auto N25 = (q[0] - 1) * (q[1] - 1) * (q[1] + 1) * (q[2] - 1) * (q[2] + 1) * q[0] / 2;
        const auto N26 =
            -(q[0] - 1) * (q[0] + 1) * (q[1] - 1) * (q[1] + 1) * (q[2] - 1) * (q[2] + 1);

        return {N0, N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15, N16, N17, N18, N19, N20, N21, N22, N23, N24, N25, N26};
    }

    static constexpr sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real> gradientShapeFunctions(const sofa::type::Vec<TopologicalDimension, Real>& q)
    {
        const type::VecNoInit N_0 = {(2 * q[0] - 1) * (q[1] - 1) * (q[2] - 1) * q[1] * q[2] / 8,
                                     (q[0] - 1) * (2 * q[1] - 1) * (q[2] - 1) * q[0] * q[2] / 8,
                                     (q[0] - 1) * (q[1] - 1) * (2 * q[2] - 1) * q[0] * q[1] / 8};
        const type::VecNoInit N_1 = {(2 * q[0] + 1) * (q[1] - 1) * (q[2] - 1) * q[1] * q[2] / 8,
                                     (q[0] + 1) * (2 * q[1] - 1) * (q[2] - 1) * q[0] * q[2] / 8,
                                     (q[0] + 1) * (q[1] - 1) * (2 * q[2] - 1) * q[0] * q[1] / 8};
        const type::VecNoInit N_2 = {(2 * q[0] + 1) * (q[1] + 1) * (q[2] - 1) * q[1] * q[2] / 8,
                                     (q[0] + 1) * (2 * q[1] + 1) * (q[2] - 1) * q[0] * q[2] / 8,
                                     (q[0] + 1) * (q[1] + 1) * (2 * q[2] - 1) * q[0] * q[1] / 8};
        const type::VecNoInit N_3 = {(2 * q[0] - 1) * (q[1] + 1) * (q[2] - 1) * q[1] * q[2] / 8,
                                     (q[0] - 1) * (2 * q[1] + 1) * (q[2] - 1) * q[0] * q[2] / 8,
                                     (q[0] - 1) * (q[1] + 1) * (2 * q[2] - 1) * q[0] * q[1] / 8};
        const type::VecNoInit N_4 = {(2 * q[0] - 1) * (q[1] - 1) * (q[2] + 1) * q[1] * q[2] / 8,
                                     (q[0] - 1) * (2 * q[1] - 1) * (q[2] + 1) * q[0] * q[2] / 8,
                                     (q[0] - 1) * (q[1] - 1) * (2 * q[2] + 1) * q[0] * q[1] / 8};
        const type::VecNoInit N_5 = {(2 * q[0] + 1) * (q[1] - 1) * (q[2] + 1) * q[1] * q[2] / 8,
                                     (q[0] + 1) * (2 * q[1] - 1) * (q[2] + 1) * q[0] * q[2] / 8,
                                     (q[0] + 1) * (q[1] - 1) * (2 * q[2] + 1) * q[0] * q[1] / 8};
        const type::VecNoInit N_6 = {(2 * q[0] + 1) * (q[1] + 1) * (q[2] + 1) * q[1] * q[2] / 8,
                                     (q[0] + 1) * (2 * q[1] + 1) * (q[2] + 1) * q[0] * q[2] / 8,
                                     (q[0] + 1) * (q[1] + 1) * (2 * q[2] + 1) * q[0] * q[1] / 8};
        const type::VecNoInit N_7 = {(2 * q[0] - 1) * (q[1] + 1) * (q[2] + 1) * q[1] * q[2] / 8,
                                     (q[0] - 1) * (2 * q[1] + 1) * (q[2] + 1) * q[0] * q[2] / 8,
                                     (q[0] - 1) * (q[1] + 1) * (2 * q[2] + 1) * q[0] * q[1] / 8};
        const type::VecNoInit N_8 = {
            -(q[1] - 1) * (q[2] - 1) * q[0] * q[1] * q[2] / 2,
            -(q[0] - 1) * (q[0] + 1) * (2 * q[1] - 1) * (q[2] - 1) * q[2] / 4,
            -(q[0] - 1) * (q[0] + 1) * (q[1] - 1) * (2 * q[2] - 1) * q[1] / 4};
        const type::VecNoInit N_9 = {
            -(2 * q[0] - 1) * (q[1] - 1) * (q[1] + 1) * (q[2] - 1) * q[2] / 4,
            -(q[0] - 1) * (q[2] - 1) * q[0] * q[1] * q[2] / 2,
            -(q[0] - 1) * (q[1] - 1) * (q[1] + 1) * (2 * q[2] - 1) * q[0] / 4};
        const type::VecNoInit N_10 = {
            -(2 * q[0] - 1) * (q[1] - 1) * (q[2] - 1) * (q[2] + 1) * q[1] / 4,
            -(q[0] - 1) * (2 * q[1] - 1) * (q[2] - 1) * (q[2] + 1) * q[0] / 4,
            -(q[0] - 1) * (q[1] - 1) * q[0] * q[1] * q[2] / 2};
        const type::VecNoInit N_11 = {
            -(2 * q[0] + 1) * (q[1] - 1) * (q[1] + 1) * (q[2] - 1) * q[2] / 4,
            -(q[0] + 1) * (q[2] - 1) * q[0] * q[1] * q[2] / 2,
            -(q[0] + 1) * (q[1] - 1) * (q[1] + 1) * (2 * q[2] - 1) * q[0] / 4};
        const type::VecNoInit N_12 = {
            -(2 * q[0] + 1) * (q[1] - 1) * (q[2] - 1) * (q[2] + 1) * q[1] / 4,
            -(q[0] + 1) * (2 * q[1] - 1) * (q[2] - 1) * (q[2] + 1) * q[0] / 4,
            -(q[0] + 1) * (q[1] - 1) * q[0] * q[1] * q[2] / 2};
        const type::VecNoInit N_13 = {
            -(q[1] + 1) * (q[2] - 1) * q[0] * q[1] * q[2] / 2,
            -(q[0] - 1) * (q[0] + 1) * (2 * q[1] + 1) * (q[2] - 1) * q[2] / 4,
            -(q[0] - 1) * (q[0] + 1) * (q[1] + 1) * (2 * q[2] - 1) * q[1] / 4};
        const type::VecNoInit N_14 = {
            -(2 * q[0] + 1) * (q[1] + 1) * (q[2] - 1) * (q[2] + 1) * q[1] / 4,
            -(q[0] + 1) * (2 * q[1] + 1) * (q[2] - 1) * (q[2] + 1) * q[0] / 4,
            -(q[0] + 1) * (q[1] + 1) * q[0] * q[1] * q[2] / 2};
        const type::VecNoInit N_15 = {
            -(2 * q[0] - 1) * (q[1] + 1) * (q[2] - 1) * (q[2] + 1) * q[1] / 4,
            -(q[0] - 1) * (2 * q[1] + 1) * (q[2] - 1) * (q[2] + 1) * q[0] / 4,
            -(q[0] - 1) * (q[1] + 1) * q[0] * q[1] * q[2] / 2};
        const type::VecNoInit N_16 = {
            -(q[1] - 1) * (q[2] + 1) * q[0] * q[1] * q[2] / 2,
            -(q[0] - 1) * (q[0] + 1) * (2 * q[1] - 1) * (q[2] + 1) * q[2] / 4,
            -(q[0] - 1) * (q[0] + 1) * (q[1] - 1) * (2 * q[2] + 1) * q[1] / 4};
        const type::VecNoInit N_17 = {
            -(2 * q[0] - 1) * (q[1] - 1) * (q[1] + 1) * (q[2] + 1) * q[2] / 4,
            -(q[0] - 1) * (q[2] + 1) * q[0] * q[1] * q[2] / 2,
            -(q[0] - 1) * (q[1] - 1) * (q[1] + 1) * (2 * q[2] + 1) * q[0] / 4};
        const type::VecNoInit N_18 = {
            -(2 * q[0] + 1) * (q[1] - 1) * (q[1] + 1) * (q[2] + 1) * q[2] / 4,
            -(q[0] + 1) * (q[2] + 1) * q[0] * q[1] * q[2] / 2,
            -(q[0] + 1) * (q[1] - 1) * (q[1] + 1) * (2 * q[2] + 1) * q[0] / 4};
        const type::VecNoInit N_19 = {
            -(q[1] + 1) * (q[2] + 1) * q[0] * q[1] * q[2] / 2,
            -(q[0] - 1) * (q[0] + 1) * (2 * q[1] + 1) * (q[2] + 1) * q[2] / 4,
            -(q[0] - 1) * (q[0] + 1) * (q[1] + 1) * (2 * q[2] + 1) * q[1] / 4};
        const type::VecNoInit N_20 = {
            (q[1] - 1) * (q[1] + 1) * (q[2] - 1) * q[0] * q[2],
            (q[0] - 1) * (q[0] + 1) * (q[2] - 1) * q[1] * q[2],
            (q[0] - 1) * (q[0] + 1) * (q[1] - 1) * (q[1] + 1) * (2 * q[2] - 1) / 2};
        const type::VecNoInit N_21 = {
            (q[1] - 1) * (q[1] + 1) * (q[2] + 1) * q[0] * q[2],
            (q[0] - 1) * (q[0] + 1) * (q[2] + 1) * q[1] * q[2],
            (q[0] - 1) * (q[0] + 1) * (q[1] - 1) * (q[1] + 1) * (2 * q[2] + 1) / 2};
        const type::VecNoInit N_22 = {
            (q[1] - 1) * (q[2] - 1) * (q[2] + 1) * q[0] * q[1],
            (q[0] - 1) * (q[0] + 1) * (2 * q[1] - 1) * (q[2] - 1) * (q[2] + 1) / 2,
            (q[0] - 1) * (q[0] + 1) * (q[1] - 1) * q[1] * q[2]};
        const type::VecNoInit N_23 = {
            (2 * q[0] + 1) * (q[1] - 1) * (q[1] + 1) * (q[2] - 1) * (q[2] + 1) / 2,
            (q[0] + 1) * (q[2] - 1) * (q[2] + 1) * q[0] * q[1],
            (q[0] + 1) * (q[1] - 1) * (q[1] + 1) * q[0] * q[2]};
        const type::VecNoInit N_24 = {
            (q[1] + 1) * (q[2] - 1) * (q[2] + 1) * q[0] * q[1],
            (q[0] - 1) * (q[0] + 1) * (2 * q[1] + 1) * (q[2] - 1) * (q[2] + 1) / 2,
            (q[0] - 1) * (q[0] + 1) * (q[1] + 1) * q[1] * q[2]};
        const type::VecNoInit N_25 = {
            (2 * q[0] - 1) * (q[1] - 1) * (q[1] + 1) * (q[2] - 1) * (q[2] + 1) / 2,
            (q[0] - 1) * (q[2] - 1) * (q[2] + 1) * q[0] * q[1],
            (q[0] - 1) * (q[1] - 1) * (q[1] + 1) * q[0] * q[2]};
        const type::VecNoInit N_26 = {
            -2 * (q[1] - 1) * (q[1] + 1) * (q[2] - 1) * (q[2] + 1) * q[0],
            -2 * (q[0] - 1) * (q[0] + 1) * (q[2] - 1) * (q[2] + 1) * q[1],
            -2 * (q[0] - 1) * (q[0] + 1) * (q[1] - 1) * (q[1] + 1) * q[2]};

        return sofa::type::Mat<NumberOfNodesInElement, TopologicalDimension, Real>(
            N_0, N_1, N_2, N_3, N_4, N_5, N_6, N_7, N_8, N_9, N_10, N_11, N_12, N_13, N_14, N_15,
            N_16, N_17, N_18, N_19, N_20, N_21, N_22, N_23, N_24, N_25, N_26);
    }

    static constexpr std::array<QuadraturePointAndWeight, 27> quadraturePoints()
    {
        constexpr Real a = 0.7745966692414834; // sqrt(3/5)
        constexpr Real w_a = 5.0 / 9.0;
        constexpr Real w_0 = 8.0 / 9.0;

        constexpr std::array<Real, 3> pts = {-a, 0.0, a};
        constexpr std::array<Real, 3> wts = {w_a, w_0, w_a};

        std::array<QuadraturePointAndWeight, 27> q {};
        int index = 0;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                for (int k = 0; k < 3; ++k)
                {
                    q[index++] = std::make_pair(
                        sofa::type::Vec<TopologicalDimension, Real>(pts[i], pts[j], pts[k]),
                        wts[i] * wts[j] * wts[k]
                    );
                }
            }
        }
        return q;
    }
};

#if !defined(SOFA_FEM_FINITE_ELEMENT_QUADRATIC_HEXAHEDRON_CPP)
extern template struct SOFA_FEM_API FiniteElement<sofa::geometry::QuadraticHexahedron, sofa::defaulttype::Vec3Types>;
#endif

}
