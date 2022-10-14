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
#include <MultiThreading/MeanComputation.h>

namespace sofa
{
TEST(MeanComputation, getTemplateName)
{
    {
        const auto engine = sofa::core::objectmodel::New<
            sofa::component::engine::MeanComputation<sofa::defaulttype::Vec3Types>
        >();
        EXPECT_EQ(engine->getTemplateName(), "Vec3d");
    }
    {
        const auto engine = sofa::core::objectmodel::New<
            sofa::component::engine::MeanComputation<sofa::defaulttype::Vec2Types>
        >();
        EXPECT_EQ(engine->getTemplateName(), "Vec2d");
    }
    {
        const auto engine = sofa::core::objectmodel::New<
            sofa::component::engine::MeanComputation<sofa::defaulttype::Vec1Types>
        >();
        EXPECT_EQ(engine->getTemplateName(), "Vec1d");
    }
    {
        const auto engine = sofa::core::objectmodel::New<
            sofa::component::engine::MeanComputation<sofa::defaulttype::Rigid2Types>
        >();
        EXPECT_EQ(engine->getTemplateName(), "Rigid2d");
    }
    {
        const auto engine = sofa::core::objectmodel::New<
            sofa::component::engine::MeanComputation<sofa::defaulttype::Rigid3Types>
        >();
        EXPECT_EQ(engine->getTemplateName(), "Rigid3d");
    }
}
}
