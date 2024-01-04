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

#include <sofa/topology/Hexahedron.h>

#include <sofa/type/Vec.h>

#include <gtest/gtest.h>

namespace sofa
{
    //from nine_hexa.msh
    static const std::array<sofa::type::Vec3d, 32> nine_hexa_vertices {{
    { 0., 0., 0. },
    { 1., 0., 0. },
    { 2., 0., 0. },
    { 3., 0., 0. },
    { 0., 1., 0. },
    { 1., 1., 0. },
    { 2., 1., 0. },
    { 3., 1., 0. },
    { 0., 2., 0. },
    { 1., 2., 0. },
    { 2., 2., 0. },
    { 3., 2., 0. },
    { 0., 3., 0. },
    { 1., 3., 0. },
    { 2., 3., 0. },
    { 3., 3., 0. },
    { 0., 0., 1. },
    { 1., 0., 1. },
    { 2., 0., 1. },
    { 3., 0., 1. },
    { 0., 1., 1. },
    { 1., 1., 1. },
    { 2., 1., 1. },
    { 3., 1., 1. },
    { 0., 2., 1. },
    { 1., 2., 1. },
    { 2., 2., 1. },
    { 3., 2., 1. },
    { 0., 3., 1. },
    { 1., 3., 1. },
    { 2., 3., 1. },
    { 3., 3., 1. }
    } };
    static const std::vector<sofa::topology::Hexahedron> nine_hexa_indices{ {
        { 0, 1, 5, 4, 16, 17, 21, 20 },
        { 1, 2, 6, 5, 17, 18, 22, 21 },
        { 2, 3, 7, 6, 18, 19, 23, 22 },
        { 4, 5, 9, 8, 20, 21, 25, 24 },
        { 5, 6, 10, 9, 21, 22, 26, 25 },
        { 6, 7, 11, 10, 23, 23 ,27 ,26 },
        { 8, 9, 13, 12, 24, 25, 29, 28 },
        { 9, 10, 14, 13, 25, 26, 30, 29},
        { 10, 11, 15, 14, 26, 27, 31, 30},
    } };

TEST(TopologyHexahedron_test, getClosestHexahedronIndex)
{   
    type::Vec3 coeffs{};
    SReal distance{};

    const sofa::type::Vec3d pos0{0.001, 0., 0.};
    EXPECT_EQ(0, sofa::topology::getClosestHexahedronIndex(nine_hexa_vertices, nine_hexa_indices, pos0,coeffs,distance));

    const sofa::type::Vec3d pos1{ 3., 3., 1.001 };
    EXPECT_EQ(8, sofa::topology::getClosestHexahedronIndex(nine_hexa_vertices, nine_hexa_indices, pos1, coeffs, distance));

    const sofa::type::Vec3d pos2{ 1.5, 1.5, 0.5 };
    EXPECT_EQ(4, sofa::topology::getClosestHexahedronIndex(nine_hexa_vertices, nine_hexa_indices, pos2, coeffs, distance));
}

}// namespace sofa
