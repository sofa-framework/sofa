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
#include <sofa/fem/FiniteElement[all].h>

namespace sofa
{

/**
 * Computes the sum of the quadrature weights and compare it to an expected value
 */
template <class ElementType, class DataTypes>
void testSumWeights(const sofa::Real_t<DataTypes> expected)
{
    using FE = sofa::fem::FiniteElement<ElementType, DataTypes>;

    SReal weightSum = 0;
    for (const auto& [q, w] : FE::quadraturePoints())
    {
        weightSum += w;
    }

    EXPECT_DOUBLE_EQ(weightSum, expected);
}

TEST(FiniteElement, edge1dWeights)
{
    testSumWeights<sofa::geometry::Edge, sofa::defaulttype::Vec1Types>(2);
}
TEST(FiniteElement, edge2dWeights)
{
    testSumWeights<sofa::geometry::Edge, sofa::defaulttype::Vec2Types>(2);
}
TEST(FiniteElement, edge3dWeights)
{
    testSumWeights<sofa::geometry::Edge, sofa::defaulttype::Vec3Types>(2);
}

TEST(FiniteElement, triangle2dWeights)
{
    testSumWeights<sofa::geometry::Triangle, sofa::defaulttype::Vec2Types>(0.5);
}
TEST(FiniteElement, triangle3dWeights)
{
    testSumWeights<sofa::geometry::Triangle, sofa::defaulttype::Vec3Types>(0.5);
}

TEST(FiniteElement, quad2dWeights)
{
    testSumWeights<sofa::geometry::Quad, sofa::defaulttype::Vec2Types>(4);
}
TEST(FiniteElement, quad3dWeights)
{
    testSumWeights<sofa::geometry::Quad, sofa::defaulttype::Vec3Types>(4);
}

TEST(FiniteElement, tetra3dWeights)
{
    testSumWeights<sofa::geometry::Tetrahedron, sofa::defaulttype::Vec3Types>(1 / 6.);
}

TEST(FiniteElement, hexa3dWeights)
{
    testSumWeights<sofa::geometry::Hexahedron, sofa::defaulttype::Vec3Types>(8);
}

/**
 * Checks that the sum of the gradients of shape functions is zero at the evaluation point.
 */
template <class ElementType, class DataTypes>
void testGradientShapeFunctions(const sofa::type::Vec<sofa::fem::FiniteElement<ElementType, DataTypes>::TopologicalDimension, sofa::Real_t<DataTypes>>& evaluationPoint)
{
    using FE = sofa::fem::FiniteElement<ElementType, DataTypes>;
    static constexpr sofa::type::Vec<FE::TopologicalDimension, sofa::Real_t<DataTypes>> zero;

    const auto N = FE::gradientShapeFunctions(evaluationPoint);

    sofa::type::Vec<FE::NumberOfNodesInElement, SReal> ones;
    std::fill(ones.begin(), ones.end(), 1);

    const auto sum = N.transposed() * ones; //compute the sum of the gradients of all shape functions
    EXPECT_EQ(sum, zero);
}

TEST(FiniteElement, edge1dGradientShapeFunctions)
{
    testGradientShapeFunctions<sofa::geometry::Edge, sofa::defaulttype::Vec1Types>(sofa::type::Vec1(0.));
    testGradientShapeFunctions<sofa::geometry::Edge, sofa::defaulttype::Vec1Types>(sofa::type::Vec1(1.));
    testGradientShapeFunctions<sofa::geometry::Edge, sofa::defaulttype::Vec1Types>(sofa::type::Vec1(-1.));
}

TEST(FiniteElement, edge2dGradientShapeFunctions)
{
    testGradientShapeFunctions<sofa::geometry::Edge, sofa::defaulttype::Vec2Types>(sofa::type::Vec1(0.));
    testGradientShapeFunctions<sofa::geometry::Edge, sofa::defaulttype::Vec2Types>(sofa::type::Vec1(1.));
    testGradientShapeFunctions<sofa::geometry::Edge, sofa::defaulttype::Vec2Types>(sofa::type::Vec1(-1.));
}

TEST(FiniteElement, edge3dGradientShapeFunctions)
{
    testGradientShapeFunctions<sofa::geometry::Edge, sofa::defaulttype::Vec3Types>(sofa::type::Vec1(0.));
    testGradientShapeFunctions<sofa::geometry::Edge, sofa::defaulttype::Vec3Types>(sofa::type::Vec1(1.));
    testGradientShapeFunctions<sofa::geometry::Edge, sofa::defaulttype::Vec3Types>(sofa::type::Vec1(-1.));
}

TEST(FiniteElement, triangle2dGradientShapeFunctions)
{
    testGradientShapeFunctions<sofa::geometry::Triangle, sofa::defaulttype::Vec2Types>(sofa::type::Vec2(0., 0.));
    testGradientShapeFunctions<sofa::geometry::Triangle, sofa::defaulttype::Vec2Types>(sofa::type::Vec2(1., 1.));
}

TEST(FiniteElement, triangle3dGradientShapeFunctions)
{
    testGradientShapeFunctions<sofa::geometry::Triangle, sofa::defaulttype::Vec3Types>(sofa::type::Vec2(0., 0.));
    testGradientShapeFunctions<sofa::geometry::Triangle, sofa::defaulttype::Vec3Types>(sofa::type::Vec2(1., 1.));
}

TEST(FiniteElement, quad2dGradientShapeFunctions)
{
    testGradientShapeFunctions<sofa::geometry::Quad, sofa::defaulttype::Vec2Types>(sofa::type::Vec2(0., 0.));
    testGradientShapeFunctions<sofa::geometry::Quad, sofa::defaulttype::Vec2Types>(sofa::type::Vec2(1., 1.));
}

TEST(FiniteElement, quad3dGradientShapeFunctions)
{
    testGradientShapeFunctions<sofa::geometry::Quad, sofa::defaulttype::Vec3Types>(sofa::type::Vec2(0., 0.));
    testGradientShapeFunctions<sofa::geometry::Quad, sofa::defaulttype::Vec3Types>(sofa::type::Vec2(1., 1.));
}

TEST(FiniteElement, tetra3dGradientShapeFunctions)
{
    testGradientShapeFunctions<sofa::geometry::Tetrahedron, sofa::defaulttype::Vec3Types>(sofa::type::Vec3(0., 0., 0.));
    testGradientShapeFunctions<sofa::geometry::Tetrahedron, sofa::defaulttype::Vec3Types>(sofa::type::Vec3(1., 1., 1.));
}

TEST(FiniteElement, hexa3dGradientShapeFunctions)
{
    testGradientShapeFunctions<sofa::geometry::Hexahedron, sofa::defaulttype::Vec3Types>(sofa::type::Vec3(0., 0., 0.));
    testGradientShapeFunctions<sofa::geometry::Hexahedron, sofa::defaulttype::Vec3Types>(sofa::type::Vec3(1., 1., 1.));
}



}
