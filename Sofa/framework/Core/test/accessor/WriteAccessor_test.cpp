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

#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/DataCallback.h>
#include <sofa/type/vector.h>

namespace sofa
{

TEST(WriteAccessor, PrimitiveTypes)
{
    Data<float> float_value { 12.f };
    sofa::helper::WriteAccessor float_accessor(float_value);
    EXPECT_FLOAT_EQ(float_accessor.ref(), 12.f);
    float_accessor.wref() = 14.f;
    EXPECT_FLOAT_EQ(float_accessor.ref(), 14.f);
    EXPECT_FLOAT_EQ(float_accessor, 14.f);
    EXPECT_FLOAT_EQ(float_value.getValue(), 14.f);

    Data<std::size_t> size_t_value { 8 };
    sofa::helper::WriteAccessor size_t_accessor(size_t_value);
    EXPECT_EQ(size_t_accessor.ref(), 8);
    size_t_accessor.wref() = 9;
    EXPECT_EQ(size_t_accessor.ref(), 9);
    EXPECT_EQ(size_t_accessor, 9);
    EXPECT_EQ(size_t_value.getValue(), 9);
}

TEST(WriteAccessor, VectorTypes)
{
    Data<sofa::type::vector<float> > vector { sofa::type::vector<float> { 0.f, 1.f, 2.f, 3.f, 4.f} };
    sofa::helper::WriteAccessor accessor(vector);

    EXPECT_EQ(accessor.size(), vector.getValue().size());
    EXPECT_EQ(accessor.size(), 5);
    EXPECT_EQ(accessor.empty(), vector.getValue().empty());
    EXPECT_EQ(accessor.begin(), vector.getValue().begin());
    EXPECT_EQ(accessor.end(), vector.getValue().end());

    for(auto& v : accessor)
    {
        ++v;
    }

    EXPECT_FLOAT_EQ(vector.getValue()[0], 1.f);
    EXPECT_FLOAT_EQ(vector.getValue()[1], 2.f);
    EXPECT_FLOAT_EQ(vector.getValue()[2], 3.f);
    EXPECT_FLOAT_EQ(vector.getValue()[3], 4.f);
    EXPECT_FLOAT_EQ(vector.getValue()[4], 5.f);

    EXPECT_FLOAT_EQ(accessor[0], 1.f);
    EXPECT_FLOAT_EQ(accessor[1], 2.f);
    EXPECT_FLOAT_EQ(accessor[2], 3.f);
    EXPECT_FLOAT_EQ(accessor[3], 4.f);
    EXPECT_FLOAT_EQ(accessor[4], 5.f);

    accessor[3] = 6.f;
    EXPECT_FLOAT_EQ(accessor[3], 6.f);
    EXPECT_FLOAT_EQ(vector.getValue()[3], 6.f);
}

TEST(WriteAccessor, MoveConstructor)
{
    Data<float> floatValue { 12.f };
    int initialCounter = floatValue.getCounter();

    {
        sofa::helper::WriteAccessor floatAccessor(floatValue);
        EXPECT_EQ(floatValue.getCounter(), initialCounter + 1);
        EXPECT_FLOAT_EQ(floatAccessor.ref(), 12.f);

        sofa::helper::WriteAccessor floatAccessorMoved(std::move(floatAccessor));
        EXPECT_EQ(floatValue.getCounter(), initialCounter + 1);
        EXPECT_FLOAT_EQ(floatAccessorMoved.ref(), 12.f);

        // Even if we moved from it, we can still call ref() and wref()
        // but it is not recommended as it doesn't hold the responsibility of endEdit anymore.
        EXPECT_FLOAT_EQ(floatAccessor.ref(), 12.f);

        floatAccessorMoved.wref() = 14.f;
        EXPECT_FLOAT_EQ(floatValue.getValue(), 14.f);
    }

    Data<float> data { 1.f };
    int endEditCount = 0;
    sofa::core::objectmodel::DataCallback cb;
    cb.addInput(&data);
    cb.addCallback([&endEditCount](){
        endEditCount++;
    });

    {
        sofa::helper::WriteAccessor acc1(data);
        EXPECT_EQ(endEditCount, 0); // beginEdit doesn't trigger callback
        {
            sofa::helper::WriteAccessor acc2(std::move(acc1));
            EXPECT_EQ(endEditCount, 0);
        }
        // acc2 out of scope, endEdit called
        EXPECT_EQ(endEditCount, 1);
    }
    // acc1 out of scope, if move constructor worked correctly, endEdit should NOT be called again.
    EXPECT_EQ(endEditCount, 1);
}

TEST(WriteAccessor, MoveConstructorVector)
{
    Data<sofa::type::vector<float>> vector { sofa::type::vector<float> { 0.f, 1.f, 2.f, 3.f, 4.f} };
    int endEditCount = 0;
    sofa::core::objectmodel::DataCallback cb;
    cb.addInput(&vector);
    cb.addCallback([&endEditCount](){
        endEditCount++;
    });

    {
        sofa::helper::WriteAccessor acc1(vector);
        EXPECT_EQ(acc1.size(), 5);
        {
            sofa::helper::WriteAccessor acc2(std::move(acc1));
            EXPECT_EQ(acc2.size(), 5);
            EXPECT_EQ(acc1.size(), 5); // Still valid but no longer responsible for endEdit
            acc2[0] = 10.f;
        }
        EXPECT_EQ(endEditCount, 1);
    }
    EXPECT_EQ(endEditCount, 1);
    EXPECT_FLOAT_EQ(vector.getValue()[0], 10.f);
}


}
