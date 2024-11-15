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
#include <sofa/component/mapping/nonlinear/VolumeMapping.h>
#include <sofa/component/mapping/testing/MappingTestCreation.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyContainer.h>

namespace sofa
{

SReal computeVolume(const sofa::type::fixed_array<sofa::type::Vec3, 4>& vertices)
{
    const auto a = vertices[1] - vertices[0];
    const auto b = vertices[2] - vertices[0];
    const auto c = vertices[3] - vertices[0];

    return std::abs(sofa::type::dot(a, sofa::type::cross(b, c))) / 6;
};

sofa::type::Mat<4,3,SReal> computeDerivativeVolume(const sofa::type::fixed_array<sofa::type::Vec3, 4>& vertices)
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

    const sofa::type::Mat<4,3,SReal> dV = computeDerivativeVolume(vertices);

    for (unsigned int vId = 0; vId < 4; ++vId)
    {
        for (unsigned int axis = 0; axis < 3; ++axis)
        {
            auto perturbation = vertices;
            perturbation[vId][axis] += small_step;

            const auto volumePlus = computeVolume(perturbation);

            perturbation = vertices;
            perturbation[vId][axis] -= small_step;

            const auto volumeMinus = computeVolume(perturbation);

            const SReal centralDifference = (volumePlus - volumeMinus) / (2 * small_step);

            EXPECT_NEAR(centralDifference, dV[vId][axis], 1e-3) << "vId = " << vId << ", i = " << axis;
        }
    }
}

TEST(VolumeMapping, secondDerivative)
{
    constexpr type::fixed_array<sofa::type::Vec3, 4> vertices{
        sofa::type::Vec3{43, -432, 1},
        sofa::type::Vec3{-53, 85, 32},
        sofa::type::Vec3{874, -413, -3},
        sofa::type::Vec3{-76, 3, -12}
    };

    const auto dV2 = component::mapping::nonlinear::VolumeMapping<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec1Types>::computeSecondDerivativeVolume(vertices);

    for (unsigned int i = 0; i < 4; ++i)
    {
        for (unsigned int j = 0; j < 4; ++j)
        {
            sofa::type::Mat<3,3,SReal> d2Volume_dPidPj_T;

            for (unsigned int axis = 0; axis < 3; ++axis)
            {
                auto perturbation = vertices;

                perturbation[j][axis] += small_step;

                const auto derivativePlus = computeDerivativeVolume(perturbation)[i];

                perturbation = vertices;
                perturbation[j][axis] -= small_step;
                const auto derivativeMinus = computeDerivativeVolume(perturbation)[i];

                const auto centralDifference = (derivativePlus - derivativeMinus) / (2 * small_step);

                d2Volume_dPidPj_T[axis] = centralDifference;
            }
            const auto d2Volume_dPidPj = d2Volume_dPidPj_T.transposed();

            for (unsigned int p = 0; p < 3; ++p)
            {
                for (unsigned int q = 0; q < 3; ++q)
                {
                    EXPECT_NEAR(d2Volume_dPidPj[p][q], dV2[i][j][p][q], 1e-3) << "i = " << i << ", j = " << j << ", p = " << p << ", q = " << q << "\n" << d2Volume_dPidPj;
                }
            }

        }
    }
}


/**
 * Test suite for VolumeMapping.
 */
template <typename VolumeMapping>
struct VolumeMappingTest : public mapping_test::Mapping_test<VolumeMapping>
{
    using In = typename VolumeMapping::In;
    using Out = typename VolumeMapping::Out;

    bool test()
    {
        VolumeMapping* map = static_cast<VolumeMapping*>( this->mapping );
        sofa::helper::getWriteAccessor(map->d_geometricStiffness)->setSelectedItem(1);

        const auto tetrahedra = sofa::core::objectmodel::New<component::topology::container::dynamic::TetrahedronSetTopologyContainer>();
        this->root->addObject(tetrahedra);
        tetrahedra->addTetra(0, 1, 2, 3);

        // parent positions
        VecCoord_t<In> incoord(4);
        In::set( incoord[0], 0,0,0 );
        In::set( incoord[1], 1,0,0 );
        In::set( incoord[2], 0,1,0 );
        In::set( incoord[3], 0,0,1 );

        // expected child positions
        VecCoord_t<Out> expectedoutcoord;
        expectedoutcoord.emplace_back( 1 / 6_sreal );

        return this->runTest( incoord, expectedoutcoord );
    }

};


using ::testing::Types;
typedef Types<
    component::mapping::nonlinear::VolumeMapping<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec1Types>
> DataTypes;

TYPED_TEST_SUITE( VolumeMappingTest, DataTypes );

// test case
TYPED_TEST( VolumeMappingTest , test )
{
    ASSERT_TRUE(this->test());
}

}
