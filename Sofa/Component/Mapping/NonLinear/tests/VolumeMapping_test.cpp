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
#include <gtest/gtest.h>
#include <sofa/type/Mat.h>


namespace sofa
{

SReal computeVolume(const sofa::type::fixed_array<sofa::type::Vec3, 4>& vertices)
{
    const auto a = vertices[1] - vertices[0];
    const auto b = vertices[2] - vertices[0];
    const auto c = vertices[3] - vertices[0];

    return std::abs(sofa::type::dot(a, sofa::type::cross(b, c))) / 6;
};

sofa::type::Mat<4,3,SReal> computeDerivativeArea(const sofa::type::fixed_array<sofa::type::Vec3, 4>& vertices)
{
    const sofa::type::fixed_array<sofa::type::Vec3, 3> v {
        vertices[1] - vertices[0],
        vertices[2] - vertices[0],
        vertices[3] - vertices[0]
    };

    const sofa::type::fixed_array<sofa::type::Vec3, 3> c {
        sofa::type::cross(v[1], v[2]),
        sofa::type::cross(v[2], v[0]),
        sofa::type::cross(v[0], v[1])
    };

    sofa::type::Mat<4, 3, SReal> result { type::NOINIT };

    result[0] = -(c[0] + c[1] + c[2]);
    result[1] = c[0];
    result[2] = c[1];
    result[3] = c[2];

    return result / 6;
}


constexpr SReal small_step = 1e-6;

TEST(VolumeMapping, firstDerivative)
{
    constexpr type::fixed_array<sofa::type::Vec3, 4> vertices{
        sofa::type::Vec3{43, -432, 1},
        sofa::type::Vec3{-53, 85, 32},
        sofa::type::Vec3{874, -413, -3},
        sofa::type::Vec3{-76, 3, -12}
    };

    const sofa::type::Mat<4,3,SReal> dV = computeDerivativeArea(vertices);

    static constexpr SReal h = 1e-6;
    for (unsigned int vId = 0; vId < 4; ++vId)
    {
        for (unsigned int axis = 0; axis < 3; ++axis)
        {
            auto perturbation = vertices;
            perturbation[vId][axis] += small_step;

            const auto areaPlus = computeVolume(perturbation);

            perturbation = vertices;
            perturbation[vId][axis] -= small_step;

            const auto areaMinus = computeVolume(perturbation);

            const SReal centralDifference = (areaPlus - areaMinus) / (2 * small_step);

            EXPECT_NEAR(centralDifference, dV[vId][axis], 1e-3) << "vId = " << vId << ", i = " << axis;
        }
    }
}


}
