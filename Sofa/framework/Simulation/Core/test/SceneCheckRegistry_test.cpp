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
#include <sofa/simulation/SceneCheckRegistry.h>

namespace sofa
{

class DummySceneCheck : public sofa::simulation::SceneCheck
{
    const std::string getName() override { return "DummySceneCheck"; }
    const std::string getDesc() override { return "For tests"; }
    void doCheckOn(sofa::simulation::Node*) override {}
};

TEST(SceneCheckRegistry, addToRegistry)
{
    simulation::SceneCheckRegistry registry;

    EXPECT_TRUE(registry.getRegisteredSceneChecks().empty());

    const auto sceneCheck = std::make_shared<DummySceneCheck>();
    EXPECT_TRUE(registry.addToRegistry(sceneCheck));

    EXPECT_FALSE(registry.getRegisteredSceneChecks().empty());
    EXPECT_EQ(registry.getRegisteredSceneChecks().size(), 1);

    const auto anotherSceneCheck = std::make_shared<DummySceneCheck>();
    EXPECT_TRUE(registry.addToRegistry(anotherSceneCheck));

    EXPECT_FALSE(registry.getRegisteredSceneChecks().empty());
    EXPECT_EQ(registry.getRegisteredSceneChecks().size(), 2);

    EXPECT_FALSE(registry.addToRegistry(sceneCheck));
    EXPECT_EQ(registry.getRegisteredSceneChecks().size(), 2);

    registry.removeFromRegistry(anotherSceneCheck);
    EXPECT_EQ(registry.getRegisteredSceneChecks().size(), 1);
}
}
