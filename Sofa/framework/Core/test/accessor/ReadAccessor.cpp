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

#include <sofa/type/vector.h>
#include <sofa/core/objectmodel/Data.h>

namespace sofa
{

TEST(ReadAccessor, DataPrimitiveTypes)
{
    const Data<float> float_value { 12.f };
    const sofa::helper::ReadAccessor float_accessor(float_value);
    EXPECT_FLOAT_EQ(float_accessor.ref(), 12.f);

    const Data<std::size_t> size_t_value = 8;
    const sofa::helper::ReadAccessor size_t_accessor(size_t_value);
    EXPECT_EQ(size_t_accessor.ref(), 8);
}

TEST(ReadAccessor, DataVectorTypes)
{
    const Data<sofa::type::vector<float> > vector { sofa::type::vector<float> { 0.f, 1.f, 2.f, 3.f, 4.f} };
    const sofa::helper::ReadAccessor accessor(vector);

    EXPECT_EQ(accessor.size(), vector.getValue().size());
    EXPECT_EQ(accessor.size(), 5);
    EXPECT_EQ(accessor.empty(), vector.getValue().empty());
    EXPECT_EQ(accessor.begin(), vector.getValue().begin());
    EXPECT_EQ(accessor.end(), vector.getValue().end());
}


}
