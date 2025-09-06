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

#include <sofa/defaulttype/VecTypes.h>
#include <gtest/gtest.h>

template < sofa::Size N, typename ValueType>
void defaultConstructor()
{
    const sofa::type::Vec<N, ValueType> vec;
    for (const auto& value : vec)
    {
        EXPECT_EQ(value, ValueType());
    }
}


TEST(VecTest, DefaultConstructor)
{
    defaultConstructor<1, unsigned int>();
    defaultConstructor<2, unsigned int>();
    defaultConstructor<3, unsigned int>();
    defaultConstructor<6, unsigned int>();

    defaultConstructor<1, int>();
    defaultConstructor<2, int>();
    defaultConstructor<3, int>();
    defaultConstructor<6, int>();

    defaultConstructor<1, float>();
    defaultConstructor<2, float>();
    defaultConstructor<3, float>();
    defaultConstructor<6, float>();

    defaultConstructor<1, double>();
    defaultConstructor<2, double>();
    defaultConstructor<3, double>();
    defaultConstructor<6, double>();
}

TEST(VecTest, StructuredBindings)
{
    constexpr sofa::type::Vec3 vec { 1.0, 2.0, 3.0 };
    const auto& [a, b, c] = vec;
    EXPECT_EQ(a, 1.);
    EXPECT_EQ(b, 2.);
    EXPECT_EQ(c, 3.);
}

TEST(VecTest, DeductionGuide)
{
    constexpr sofa::type::Vec vec { 1.0_sreal, 2.0_sreal, 3.0_sreal };
    static_assert(std::is_same_v<decltype(vec)::value_type, SReal>);

    constexpr sofa::type::VecNoInit vec2 { 1.0_sreal, 2.0_sreal, 3.0_sreal };
    static_assert(std::is_same_v<decltype(vec2)::value_type, SReal>);
}

TEST(VecTest, Equality)
{
    constexpr sofa::type::Vec<3,SReal> vecf1 { 1.0_sreal, 2.0_sreal, 3.0_sreal };
    constexpr sofa::type::Vec<3,SReal> vecf2 { 1.0_sreal, 2.0_sreal, 3.0_sreal };
    constexpr sofa::type::Vec<3,SReal> vecf3 { 1.0_sreal, 2.00001_sreal, 3.0_sreal };
    
    constexpr sofa::type::Vec<3,int> veci1 { 2, 4, 7 };
    constexpr sofa::type::Vec<3,int> veci2 { 2, 4, 7 };
    constexpr sofa::type::Vec<3,int> veci3 { 2, 4, 6 };
    
    EXPECT_TRUE(vecf1 == vecf2);
    EXPECT_FALSE(vecf1 != vecf2);
    EXPECT_FALSE(vecf1 == vecf3);
    EXPECT_TRUE(vecf1 != vecf3);
    
    EXPECT_TRUE(veci1 == veci2);
    EXPECT_FALSE(veci1 != veci2);
    EXPECT_FALSE(veci1 == veci3);
    EXPECT_TRUE(veci1 != veci3);
}
