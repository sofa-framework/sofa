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

SReal computeArea(const sofa::type::Vec3& A, const sofa::type::Vec3& B, const sofa::type::Vec3& C)
{
    const auto AB = B - A;
    const auto AC = C - A;

    const auto N = sofa::type::cross(AB, AC);
    const auto n = N.norm();

    return 0.5 * n;
};


TEST(AreaMapping, crossProductDerivative)
{
    constexpr sofa::type::fixed_array<sofa::type::Vec3, 3> vertices{
        sofa::type::Vec3{43, -432, 1},
        sofa::type::Vec3{-53,85,32},
        sofa::type::Vec3{874, -413, -3}
    };

    constexpr auto AB = vertices[1] - vertices[0];
    constexpr auto AC = vertices[2] - vertices[0];

    constexpr auto N = sofa::type::cross(AB, AC);
    const auto n = sofa::type::cross(AB, AC).norm();

    constexpr auto dN_dA = -sofa::type::crossProductMatrix(vertices[2]-vertices[1]);
    constexpr auto dN_dB = -sofa::type::crossProductMatrix(vertices[0]-vertices[2]);
    constexpr auto dN_dC = -sofa::type::crossProductMatrix(vertices[1]-vertices[0]);

    const sofa::type::Mat<3,3,SReal> dA {
        sofa::type::Vec3{ 1 / (2 * n) * dN_dA * N}, // dArea_dA
        sofa::type::Vec3{ 1 / (2 * n) * dN_dB * N}, // dArea_dB
        sofa::type::Vec3{ 1 / (2 * n) * dN_dC * N}, // dArea_dC
    };

    static constexpr SReal h = 1e-6;
    for (unsigned int vId = 0; vId < 3; ++vId)
    {
        for (unsigned int dim = 0; dim < 3; ++dim)
        {
            auto perturbation = vertices;
            perturbation[vId][dim] += h;

            const auto areaPlus = computeArea(perturbation[0], perturbation[1], perturbation[2]);

            perturbation = vertices;
            perturbation[vId][dim] -= h;

            const auto areaMinus = computeArea(perturbation[0], perturbation[1], perturbation[2]);

            const SReal centralDifference = (areaPlus - areaMinus) / (2 * h);

            EXPECT_NEAR(centralDifference, dA[vId][dim], 1e-3) << "vId = " << vId << ", i = " << dim;
        }
    }
}





/**
 * Test suite for AreaMapping.
 */
template <typename AreaMapping>
struct AreaMappingTest : public mapping_test::Mapping_test<AreaMapping>
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
        expectedoutcoord.push_back( type::Vec1( std::sqrt(2.0) / 2 ) );

        return this->runTest( incoord, expectedoutcoord );
    }

};


using ::testing::Types;
typedef Types<
    component::mapping::nonlinear::AreaMapping<defaulttype::Vec3Types,defaulttype::Vec1Types>
> DataTypes;

TYPED_TEST_SUITE( AreaMappingTest, DataTypes );

// test case
TYPED_TEST( AreaMappingTest , test )
{
    this->flags &= ~AreaMappingTest<TypeParam>::TEST_getJs;
    ASSERT_TRUE(this->test());
}


}
