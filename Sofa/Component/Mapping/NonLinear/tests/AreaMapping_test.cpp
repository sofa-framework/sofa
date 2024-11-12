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
#include <sofa/component/mapping/nonlinear/AreaMapping.h>
#include <gtest/gtest.h>
#include <sofa/component/mapping/testing/MappingTestCreation.h>
#include <sofa/component/topology/container/dynamic/TriangleSetTopologyContainer.h>


namespace sofa
{
using component::mapping::nonlinear::AreaMapping;

SReal computeArea(const sofa::type::fixed_array<sofa::type::Vec3, 3>& vertices)
{
    const auto AB = vertices[1] - vertices[0];
    const auto AC = vertices[2] - vertices[0];

    const auto N = sofa::type::cross(AB, AC);
    const auto n = N.norm();

    return 0.5 * n;
};

sofa::type::Mat<3,3,SReal> computeDerivativeArea(const sofa::type::fixed_array<sofa::type::Vec3, 3>& vertices)
{
    const auto AB = vertices[1] - vertices[0];
    const auto AC = vertices[2] - vertices[0];

    const auto N = sofa::type::cross(AB, AC);
    const auto n = N.norm();

    sofa::type::Mat<3,3,SReal> result { type::NOINIT };
    for (unsigned int i = 0; i < 3; ++i)
    {
        result[i] = -(1 / (2 * n)) * (vertices[(2 + i) % 3] - vertices[(1 + i) % 3]).cross(N);
    }

    return result;
}

constexpr SReal small_step = 1e-6;

TEST(AreaMapping, firstDerivative)
{
    constexpr sofa::type::fixed_array<sofa::type::Vec3, 3> vertices{
        sofa::type::Vec3{43, -432, 1},
        sofa::type::Vec3{-53,85,32},
        sofa::type::Vec3{874, -413, -3}
    };

    const sofa::type::Mat<3,3,SReal> dA = computeDerivativeArea(vertices);

    for (unsigned int vId = 0; vId < 3; ++vId)
    {
        for (unsigned int axis = 0; axis < 3; ++axis)
        {
            auto perturbation = vertices;
            perturbation[vId][axis] += small_step;

            const auto areaPlus = computeArea(perturbation);

            perturbation = vertices;
            perturbation[vId][axis] -= small_step;

            const auto areaMinus = computeArea(perturbation);

            const SReal centralDifference = (areaPlus - areaMinus) / (2 * small_step);

            EXPECT_NEAR(centralDifference, dA[vId][axis], 1e-3) << "vId = " << vId << ", i = " << axis;
        }
    }
}


TEST(AreaMapping, secondDerivative)
{
    constexpr sofa::type::fixed_array<sofa::type::Vec3, 3> vertices{
        sofa::type::Vec3{43, -432, 1},
        sofa::type::Vec3{-53,85,32},
        sofa::type::Vec3{874, -413, -3}
    };

    const auto dA2 = AreaMapping<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec1Types>::computeSecondDerivativeArea({vertices[0], vertices[1], vertices[2]});

    for (unsigned int i = 0; i < 3; ++i)
    {
        for (unsigned int j = 0; j < 3; ++j)
        {
            sofa::type::Mat<3,3,SReal> d2Area_dPidPj;

            for (unsigned int axis = 0; axis < 3; ++axis)
            {
                auto perturbation = vertices;

                perturbation[j][axis] += small_step;

                const auto derivativePlus = computeDerivativeArea(perturbation)[i];

                perturbation = vertices;
                perturbation[j][axis] -= small_step;
                const auto derivativeMinus = computeDerivativeArea(perturbation)[i];

                const auto centralDifference = (derivativePlus - derivativeMinus) / (2 * small_step);

                d2Area_dPidPj[axis] = centralDifference;
            }
            d2Area_dPidPj.transpose();

            for (unsigned int p = 0; p < 3; ++p)
            {
                for (unsigned int q = 0; q < 3; ++q)
                {
                    EXPECT_NEAR(d2Area_dPidPj[p][q], dA2[i][j][p][q], 1e-6) << "i = " << i << ", j = " << j << ", p = " << p << ", q = " << q << "\n" << d2Area_dPidPj;
                }
            }

        }
    }
}





/**
 * Test suite for AreaMapping.
 */
template <typename AreaMapping>
struct AreaMappingTest : public sofa::mapping_test::Mapping_test<AreaMapping>
{
    using In = typename AreaMapping::In;
    using Out = typename AreaMapping::Out;

    bool test()
    {
        AreaMapping* map = static_cast<AreaMapping*>( this->mapping );
        sofa::helper::getWriteAccessor(map->d_geometricStiffness)->setSelectedItem(1);

        const auto triangles = sofa::core::objectmodel::New<component::topology::container::dynamic::TriangleSetTopologyContainer>();
        this->root->addObject(triangles);
        triangles->addTriangle(0, 1, 2);

        // parent positions
        VecCoord_t<In> incoord(3);
        In::set( incoord[0], 0,0,0 );
        In::set( incoord[1], 1,0,0 );
        In::set( incoord[2], 1,1,0 );

        // expected child positions
        VecCoord_t<Out> expectedoutcoord;
        expectedoutcoord.emplace_back( 0.5 );

        return this->runTest( incoord, expectedoutcoord );
    }

};


using ::testing::Types;
typedef Types<
    AreaMapping<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec1Types>
> DataTypes;

TYPED_TEST_SUITE( AreaMappingTest, DataTypes );

// test case
TYPED_TEST( AreaMappingTest , test )
{
    ASSERT_TRUE(this->test());
}


}
