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
#include <MultiThreading/DataExchange.h>

namespace sofa
{
TEST(DataExchange, getTemplateName)
{
    {
        const auto engine = sofa::core::objectmodel::New<
            sofa::core::DataExchange<sofa::type::vector<sofa::type::Vec3d>>
        >("", "");
        EXPECT_EQ(engine->getTemplateName(), "vector<Vec3d>");
    }
    {
        const auto engine = sofa::core::objectmodel::New<
            sofa::core::DataExchange<sofa::type::vector<sofa::type::Vec2d>>
        >("", "");
        EXPECT_EQ(engine->getTemplateName(), "vector<Vec2d>");
    }
    {
        const auto engine = sofa::core::objectmodel::New<
            sofa::core::DataExchange<sofa::type::vector<double>>
        >("", "");
        EXPECT_EQ(engine->getTemplateName(), "vector<double>");
    }
    {
        const auto engine = sofa::core::objectmodel::New<
            sofa::core::DataExchange<sofa::type::Vec3d>
        >("", "");
        EXPECT_EQ(engine->getTemplateName(), "Vec3d");
    }
    {
        const auto engine = sofa::core::objectmodel::New<
            sofa::core::DataExchange<double>
        >("", "");
        EXPECT_EQ(engine->getTemplateName(), "double");
    }



    {
        const auto engine = sofa::core::objectmodel::New<
            sofa::core::DataExchange<sofa::type::vector<sofa::type::Vec3f>>
        >("", "");
        EXPECT_EQ(engine->getTemplateName(), "vector<Vec3f>");
    }
    {
        const auto engine = sofa::core::objectmodel::New<
            sofa::core::DataExchange<sofa::type::vector<sofa::type::Vec2f>>
        >("", "");
        EXPECT_EQ(engine->getTemplateName(), "vector<Vec2f>");
    }
    {
        const auto engine = sofa::core::objectmodel::New<
            sofa::core::DataExchange<sofa::type::vector<float>>
        >("", "");
        EXPECT_EQ(engine->getTemplateName(), "vector<float>");
    }
    {
        const auto engine = sofa::core::objectmodel::New<
            sofa::core::DataExchange<sofa::type::Vec3f>
        >("", "");
        EXPECT_EQ(engine->getTemplateName(), "Vec3f");
    }
    {
        const auto engine = sofa::core::objectmodel::New<
            sofa::core::DataExchange<float>
        >("", "");
        EXPECT_EQ(engine->getTemplateName(), "float");
    }

    {
        const auto engine = sofa::core::objectmodel::New<
            sofa::core::DataExchange<sofa::type::vector<int>>
        >("", "");
        EXPECT_EQ(engine->getTemplateName(), "vector<int>");
    }
    {
        const auto engine = sofa::core::objectmodel::New<
            sofa::core::DataExchange<sofa::type::vector<unsigned int>>
        >("", "");
        EXPECT_EQ(engine->getTemplateName(), "vector<unsigned_int>");
    }
    {
        const auto engine = sofa::core::objectmodel::New<
            sofa::core::DataExchange<bool>
        >("", "");
        EXPECT_EQ(engine->getTemplateName(), "bool");
    }
}
}
