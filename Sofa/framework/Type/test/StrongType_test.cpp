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
#include <sofa/type/StrongType.h>

namespace sofa
{

TEST(StrongType, constructor)
{
    using myint = sofa::type::StrongType<int, struct _myinttag>;

    // myint a; //this is not possible as the default constructor is explicit
    constexpr myint a ( 2 );
    EXPECT_EQ(a.get(), 2);
}

TEST(StrongType, preIncrementable)
{
    using myint = sofa::type::StrongType<int, struct _preincrementable, sofa::type::functionality::PreIncrementable>;
    myint a ( 2 );
    ++a;
    EXPECT_EQ(a.get(), 3);
}

TEST(StrongType, postIncrementable)
{
    using myint = sofa::type::StrongType<int, struct _postincrementable, sofa::type::functionality::PostIncrementable>;
    myint a ( 2 );
    a++;
    EXPECT_EQ(a.get(), 3);
}

TEST(StrongType, floatAddable)
{
    using myfloat = sofa::type::StrongType<float, struct _floataddable, sofa::type::functionality::Addable>;
    static constexpr myfloat a ( 2 );
    static constexpr myfloat b ( 3 );
    EXPECT_EQ((a + b).get(), 5);
}

TEST(StrongType, floatBinarySubtractable)
{
    using myfloat = sofa::type::StrongType<float, struct _floatBinarySubstractable, sofa::type::functionality::BinarySubtractable>;
    static constexpr myfloat a ( 2 );
    static constexpr myfloat b ( 3 );
    EXPECT_EQ((a - b).get(), -1);
}

TEST(StrongType, floatUnarySubtractable)
{
    using myfloat = sofa::type::StrongType<float, struct _floatUnarySubstractable, sofa::type::functionality::UnarySubtractable>;
    static constexpr myfloat a ( 2 );
    EXPECT_EQ((-a).get(), -2);
}

TEST(StrongType, floatMultiplicable)
{
    using myfloat = sofa::type::StrongType<float, struct _floatMultiplicable, sofa::type::functionality::Multiplicable>;
    static constexpr myfloat a ( 2 );
    static constexpr myfloat b ( 3 );
    EXPECT_EQ((a * b).get(), 6);
}

}
