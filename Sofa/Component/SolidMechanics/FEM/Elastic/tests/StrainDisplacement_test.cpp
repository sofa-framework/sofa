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
#include <sofa/component/solidmechanics/fem/elastic/impl/StrainDisplacement.h>
#include <gtest/gtest.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/geometry/Tetrahedron.h>

namespace sofa
{

using namespace sofa::component::solidmechanics::fem::elastic;

TEST(StraintDisplacement, matrixVectorProduct)
{
    sofa::type::Mat<4, 3, SReal> dN_dq(sofa::type::NOINIT);
    for (std::size_t i = 0; i < 4; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            dN_dq(i, j) = (i + 4) * (j + 9);
        }
    }

    const auto B = makeStrainDisplacement<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>(dN_dq);

    sofa::type::Vec<12, SReal> v;

    for (std::size_t i = 0; i < 12; ++i)
    {
        v[i] = static_cast<SReal>(i);
    }

    const auto Bv = B * v;
    const auto expectedBv = B.B * v;

    for (std::size_t i = 0; i < 6; ++i)
    {
        EXPECT_DOUBLE_EQ(Bv[i], expectedBv[i]) << "i = " << i;
    }
}

TEST(StraintDisplacement, matrixTransposedVectorProduct)
{
    sofa::type::Mat<4, 3, SReal> dN_dq(sofa::type::NOINIT);
    for (std::size_t i = 0; i < 4; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            dN_dq(i, j) = (i + 4) * (j + 9);
        }
    }

    const auto B = makeStrainDisplacement<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>(dN_dq);

    sofa::type::Vec<6, SReal> v;
    for (std::size_t i = 0; i < 6; ++i)
    {
        v[i] = static_cast<SReal>(i);
    }

    const auto B_Tv = B.multTranspose(v);
    const auto expectedB_Tv = B.B.multTranspose(v);

    for (std::size_t i = 0; i < 12; ++i)
    {
        EXPECT_DOUBLE_EQ(B_Tv[i], expectedB_Tv[i]) << "i = " << i;
    }
}
}
