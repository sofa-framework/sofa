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
#include <sofa/component/mapping/linear/CellAveragingMapping.h>
#include <gtest/gtest.h>
#include <sofa/component/mapping/testing/MappingTestCreation.h>
#include <sofa/component/topology/container/dynamic/TriangleSetTopologyContainer.h>


namespace sofa
{


constexpr SReal small_step = 1e-6;

/****
 *
 *
 *  3 ___________ 2
 *    |  1    / |
 *    |    /    |
 *    | /    0  |
 *  0 ----------- 1
 */
TEST(CellAveragingMapping, firstDerivativeTwoConnectedTriangles)
{
    static constexpr sofa::type::fixed_array<SReal, 2> triangleValues{12.3218, -8.94317};

    static constexpr auto computeCellAverage =
        [](const sofa::type::fixed_array<SReal, 2>& values,
            sofa::Index vertexId)
    {
        if (vertexId == 0 || vertexId == 2)
        {
            return (values[0] + values[1]) / 2;
        }
        if (vertexId == 1)
        {
            return values[0];
        }
        return values[1];
    };

    const sofa::type::Mat<4, 2, SReal> derivative {
        {1_sreal / 2, 1_sreal / 2},
        {1_sreal, 0},
        {1_sreal / 2, 1_sreal / 2},
        {0, 1_sreal}
    };

    for (unsigned int v = 0; v < 4; ++v)
    {
        for (unsigned int t = 0; t < triangleValues.size(); ++t)
        {
            auto perturbation = triangleValues;
            perturbation[t] += small_step;

            const auto averagePlus = computeCellAverage(perturbation, v);

            perturbation = triangleValues;
            perturbation[t] -= small_step;

            const auto averageMinus = computeCellAverage(perturbation, v);

            const SReal centralDifference = (averagePlus - averageMinus) / (2 * small_step);

            EXPECT_NEAR(centralDifference, derivative[v][t], 1e-9) << "v = " << v << ", t = " << t;
        }
    }
}

using component::mapping::linear::CellAveragingMapping;

template <typename CellAveragingMapping>
struct CellAveragingMappingTest : public sofa::mapping_test::Mapping_test<CellAveragingMapping>
{
    using In = typename CellAveragingMapping::In;
    using Out = typename CellAveragingMapping::Out;

    bool oneTriangle()
    {
        const auto triangles = sofa::core::objectmodel::New<component::topology::container::dynamic::TriangleSetTopologyContainer>();
        this->root->addObject(triangles);
        triangles->addTriangle(0, 1, 2);

        CellAveragingMapping* map = static_cast<CellAveragingMapping*>( this->mapping );
        EXPECT_TRUE(map);

        map->l_topology = triangles;

        static constexpr Real_t<In> randomValue {12.3218 };

        // parent positions
        const VecCoord_t<In> inCoord(1, Coord_t<In>(randomValue));

        // expected child positions
        const VecCoord_t<Out> expectedOutCoord(3, Coord_t<In>(randomValue));

        return this->runTest( inCoord, expectedOutCoord );
    }

    bool twoConnectedTriangles()
    {
        const auto triangles = sofa::core::objectmodel::New<component::topology::container::dynamic::TriangleSetTopologyContainer>();
        this->root->addObject(triangles);
        triangles->addTriangle(0, 1, 2);
        triangles->addTriangle(0, 2, 3);

        CellAveragingMapping* map = static_cast<CellAveragingMapping*>( this->mapping );
        EXPECT_TRUE(map);

        map->l_topology = triangles;

        static constexpr sofa::type::fixed_array<Real_t<In>, 2> randomValues {12.3218, -8.94317 };
        static constexpr Real_t<In> average = (randomValues[0] + randomValues[1]) / 2;

        // parent positions
        VecCoord_t<In> inCoord(2);
        inCoord[0] = Coord_t<In>(randomValues[0]);
        inCoord[1] = Coord_t<In>(randomValues[1]);

        // expected child positions
        VecCoord_t<Out> expectedOutCoord(4);
        expectedOutCoord[0] = Coord_t<Out>(average);
        expectedOutCoord[1] = Coord_t<Out>(randomValues[0]);
        expectedOutCoord[2] = Coord_t<Out>(average);
        expectedOutCoord[3] = Coord_t<Out>(randomValues[1]);

        return this->runTest( inCoord, expectedOutCoord );
    }

};

using ::testing::Types;
typedef Types<
    CellAveragingMapping<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec1Types>
> DataTypes;

TYPED_TEST_SUITE( CellAveragingMappingTest, DataTypes );

// test case
TYPED_TEST( CellAveragingMappingTest , oneTriangle )
{
    this->flags |= TestFixture::TEST_applyJT_matrix;
    ASSERT_TRUE(this->oneTriangle());
}

TYPED_TEST( CellAveragingMappingTest , twoConnectedTriangles )
{
    this->flags |= TestFixture::TEST_applyJT_matrix;
    ASSERT_TRUE(this->twoConnectedTriangles());
}
}
