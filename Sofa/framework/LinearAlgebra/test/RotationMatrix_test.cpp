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
#include <sofa/linearalgebra/RotationMatrix.h>
#include <sofa/linearalgebra/SparseMatrix.h>

#include <gtest/gtest.h>

namespace sofa
{
TEST(RotationMatrix, rotateMatrix)
{
    linearalgebra::RotationMatrix<double> rot;
    rot.resize(9, 9);

    //x-axis rotation of pi/2 -> Rx
    static constexpr sofa::type::Mat3x3d Rx
    { type::Mat3x3d::Line{1, 0, 0}, type::Mat3x3d::Line{0, 0, -1}, type::Mat3x3d::Line{0, 1, 0}};
    rot.add(0, 0, Rx);

    //y-axis rotation of pi/2 -> Ry
    static constexpr sofa::type::Mat3x3d Ry
    { type::Mat3x3d::Line{0, 0, 1}, type::Mat3x3d::Line{0, 1, 0}, type::Mat3x3d::Line{-1, 0, 0}};
    rot.add(3, 3, Ry);

    //z-axis rotation of pi/2 -> Rz
    static constexpr sofa::type::Mat3x3d Rz
    { type::Mat3x3d::Line{0, -1, 0}, type::Mat3x3d::Line{1, 0, 0}, type::Mat3x3d::Line{0, 0, -1}};
    rot.add(6, 6, Rz);

    static constexpr sofa::type::fixed_array<sofa::type::Mat3x3d, 3> R { Rx, Ry, Rz};

    constexpr sofa::type::Mat M =
        []()
        {
            double entry {};
            sofa::type::MatNoInit<9, 9, double> M;
            for (sofa::Size i = 0; i < 9; ++i)
            {
                for (sofa::Size j = 0; j < 9; ++j)
                {
                    M(i, j) = entry++;
                }
            }
            return M;
        }();

    sofa::type::fixed_array<
        sofa::type::fixed_array<type::Mat3x3d, 3>, 3> Mij;
    for (sofa::Size i = 0; i < 3; ++i)
    {
        for (sofa::Size j = 0; j < 3; ++j)
        {
            M.getsub(i * 3, j * 3, Mij[i][j]);
        }
    }

    linearalgebra::SparseMatrix<double> mat;
    mat.resize(9, 9);

    for (sofa::SignedIndex i = 0; i < 9; ++i)
    {
        for (sofa::SignedIndex j = 0; j < 9; ++j)
        {
            mat.add(i, j, M[i][j]);
        }
    }

    // mat = [ A | B | C ]
    //       [ D | E | F ]
    //       [ G | H | I ]
    // where each letter is a 3x3 matrix

    linearalgebra::SparseMatrix<double> result;

    // result is = [ A * Rx | B * Ry | C * Rz ]
    //             [ D * Rx | E * Ry | F * Rz ]
    //             [ G * Rx | H * Ry | I * Rz ]

    rot.rotateMatrix(&result, &mat);

    sofa::type::MatNoInit<9, 9, double> expectedResult;
    for (sofa::Size i = 0; i < 3; ++i)
    {
        for (sofa::Size j = 0; j < 3; ++j)
        {
            expectedResult.setsub(i * 3, j * 3, Mij[i][j] * R[j]);
        }
    }

    for (sofa::SignedIndex i = 0; i < 9; ++i)
    {
        for (sofa::SignedIndex j = 0; j < 9; ++j)
        {
            EXPECT_DOUBLE_EQ(result.element(i, j), expectedResult(i, j));
        }
    }
}
}
