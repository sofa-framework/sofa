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

sofa::type::Mat<3,3,sofa::type::Mat<3,3,SReal>> computeSecondDerivativeArea(const sofa::type::fixed_array<sofa::type::Vec3, 3>& vertices)
{
    const auto AB = vertices[1] - vertices[0];
    const auto AC = vertices[2] - vertices[0];

    const auto N = sofa::type::cross(AB, AC);
    const auto n2 = sofa::type::dot(N, N);

    sofa::type::Mat<3,3,sofa::type::Mat<3,3,SReal>> d2A;

    const auto ka = 1 / (2 * std::sqrt(std::pow(n2, 3)));

    constexpr auto skewSign = type::crossProductMatrix(sofa::type::Vec<3, SReal>{1,1,1});

    for (unsigned int i = 0; i < 3; ++i)
    {
        for (unsigned int j = 0; j < 3; ++j)
        {
            auto& entry = d2A[i][j];

            const auto i1 = (i + 1) % 3;
            const auto j1 = (j + 1) % 3;
            const auto i2 = (i + 2) % 3;
            const auto j2 = (j + 2) % 3;

            const auto N_cross_Pi1Pi2 = N.cross(vertices[i1] - vertices[i2]);
            const auto N_cross_Pj1Pj2 = N.cross(vertices[j1] - vertices[j2]);

            const auto outer = sofa::type::dyad(N_cross_Pi1Pi2, N_cross_Pj1Pj2);
            static const auto& id = sofa::type::Mat<3, 3, SReal>::Identity();

            const auto dot_product = sofa::type::dot(vertices[i1] - vertices[i2], vertices[j1] - vertices[j2]);

            entry = ka * (- outer + n2 * (dot_product * id - sofa::type::dyad(vertices[j1] - vertices[j2], vertices[i1] - vertices[i2])));

            if (i != j)
            {
                const auto sign = skewSign[i][j];
                entry += sign * ka * n2 * type::crossProductMatrix(N);
            }

        }
    }

    return d2A;
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

    static constexpr SReal h = 1e-6;
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

    const auto dA2 = computeSecondDerivativeArea({vertices[0], vertices[1], vertices[2]});

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
    typedef typename AreaMapping::In InDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename InDataTypes::Coord InCoord;

    typedef typename AreaMapping::Out OutDataTypes;
    typedef typename OutDataTypes::VecCoord OutVecCoord;
    typedef typename OutDataTypes::Coord OutCoord;

    bool test()
    {
        AreaMapping* map = static_cast<AreaMapping*>( this->mapping );
        sofa::helper::getWriteAccessor(map->d_geometricStiffness)->setSelectedItem(1);

        const auto triangles = sofa::core::objectmodel::New<component::topology::container::dynamic::TriangleSetTopologyContainer>();
        this->root->addObject(triangles);
        triangles->addTriangle(0, 1, 2);

        // parent positions
        InVecCoord incoord(3);
        InDataTypes::set( incoord[0], 0,0,0 );
        InDataTypes::set( incoord[1], 1,0,0 );
        InDataTypes::set( incoord[2], 1,1,0 );

        // expected child positions
        OutVecCoord expectedoutcoord;
        expectedoutcoord.emplace_back( 0.5 );

        return this->runTest( incoord, expectedoutcoord );
    }

};


using ::testing::Types;
typedef Types<
    sofa::component::mapping::nonlinear::AreaMapping<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec1Types>
> DataTypes;

TYPED_TEST_SUITE( AreaMappingTest, DataTypes );

// test case
TYPED_TEST( AreaMappingTest , test )
{
    this->flags &= ~AreaMappingTest<TypeParam>::TEST_getJs;
    ASSERT_TRUE(this->test());
}


}
