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
#include <sofa/component/mapping/linear/IdentityMapping.h>
#include <gtest/gtest.h>
#include <sofa/component/mapping/testing/MappingTestCreation.h>
#include <sofa/testing/LinearCongruentialRandomGenerator.h>


namespace sofa
{

using component::mapping::linear::IdentityMapping;

template <typename IdentityMapping>
struct IdentityMappingTest : public sofa::mapping_test::Mapping_test<IdentityMapping>
{
    using In = typename IdentityMapping::In;
    using Out = typename IdentityMapping::Out;

    void generatePositions(VecCoord_t<In>& inCoord)
    {
        sofa::testing::LinearCongruentialRandomGenerator lcg(96547);
        for (auto& x : inCoord)
        {
            for (std::size_t i = 0; i < Coord_t<In>::total_size; ++i)
            {
                using Real = Real_t<In>;
                x[i] = lcg.generateInRange(static_cast<Real>(-1e2), static_cast<Real>(1e2));
            }
        }
    }

    bool test()
    {
        // parent positions
        VecCoord_t<In> inCoord(150);
        generatePositions(inCoord);

        // expected child positions
        VecCoord_t<Out> expectedOutCoord;
        expectedOutCoord.reserve(inCoord.size());
        for (const auto& x : inCoord)
        {
            Coord_t<Out> y;
            for (std::size_t i = 0; i < Coord_t<Out>::total_size; ++i)
            {
                y[i] = x[i];
            }
            expectedOutCoord.emplace_back(y);
        }

        return this->runTest( inCoord, expectedOutCoord );
    }

};

using ::testing::Types;
typedef Types<
    IdentityMapping<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types>,
    IdentityMapping<sofa::defaulttype::Vec2Types, sofa::defaulttype::Vec2Types>,
    IdentityMapping<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec1Types>,
    IdentityMapping<sofa::defaulttype::Vec6Types, sofa::defaulttype::Vec3Types>,
    IdentityMapping<sofa::defaulttype::Vec6Types, sofa::defaulttype::Vec6Types>,
    IdentityMapping<sofa::defaulttype::Rigid3Types, sofa::defaulttype::Rigid3Types>,
    IdentityMapping<sofa::defaulttype::Rigid2Types, sofa::defaulttype::Rigid2Types>,
    IdentityMapping<sofa::defaulttype::Rigid3Types, sofa::defaulttype::Vec3Types>,
    IdentityMapping<sofa::defaulttype::Rigid2Types, sofa::defaulttype::Vec2Types>
> DataTypes;

TYPED_TEST_SUITE( IdentityMappingTest, DataTypes );

// test case
TYPED_TEST( IdentityMappingTest , test )
{
    this->flags |= TestFixture::TEST_applyJT_matrix;
    this->errorMax = 1e2;
    ASSERT_TRUE(this->test());
}

}
