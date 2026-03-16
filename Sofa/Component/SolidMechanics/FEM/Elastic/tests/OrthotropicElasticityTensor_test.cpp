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
#include <sofa/component/solidmechanics/fem/elastic/impl/LameParameters.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/OrthotropicElasticityTensor.h>
#include <sofa/testing/NumericTest.h>

namespace sofa
{

using namespace sofa::component::solidmechanics::fem::elastic;

TEST(OrthotropicElasticityTensor, isotropicElasticityTensor)
{
    constexpr auto youngModulus = 1_sreal;
    constexpr auto poissonRatio = 0_sreal;

    LameLambda<SReal> lambda { 0 };
    LameMu<SReal> mu { 0 };
        component::solidmechanics::fem::elastic::toLameParameters<3, SReal>(
            YoungModulus<SReal>(youngModulus), PoissonRatio<SReal>(poissonRatio),
            lambda, mu);
    const auto C =
        component::solidmechanics::fem::elastic::makeIsotropicElasticityTensor<3, SReal>(mu, lambda).toVoigtMatSym();

    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            EXPECT_DOUBLE_EQ(C(i, j), static_cast<SReal>(i == j)) << "i = " << i << " j = " << j;
        }
    }

    for (std::size_t i = 3; i < 6; ++i)
    {
        for (std::size_t j = 3; j < 6; ++j)
        {
            EXPECT_FLOATINGPOINT_EQ(C(i, j), static_cast<SReal>(i == j) * 0.5_sreal);
        }
    }

    for (std::size_t i = 3; i < 6; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            EXPECT_FLOATINGPOINT_EQ(C(i, j), 0_sreal);
            EXPECT_FLOATINGPOINT_EQ(C(j, i), 0_sreal);
        }
    }
}

}
